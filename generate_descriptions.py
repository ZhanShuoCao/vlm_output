"""
generate_descriptions.py
========================
Use Qwen-VL (DashScope API) to generate per-image visual descriptions
for MVTec AD defect images, output a COCO-style JSON annotation file.

Usage:
    # Step 1: Set environment variables
    export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    export MVTEC_ROOT=/path/to/MVTec          # default: ./MVTec
    export OUTPUT_FILE=./mvtec_descriptions.json  # default

    # Step 2: Run
    python generate_descriptions.py
"""

import os
import json
import base64
import time
import sys
from pathlib import Path
from datetime import datetime

import dashscope
from dashscope import MultiModalConversation
from tqdm import tqdm

# ──────────────────── Configuration ────────────────────
MODEL = "qwen-vl-max"            # best quality; switch to qwen-vl-plus for lower cost
MODEL_FALLBACK = ["qwen-vl-max", "qwen-vl-plus", "qwen-vl-turbo"]  # auto-degrade on quota exhaustion
MVTEC_ROOT = os.environ.get("MVTEC_ROOT", "./MVTec")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "./mvtec_descriptions.json")
PROMPT_FILE = os.environ.get("PROMPT_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_prompt_mvtec_ago_template.txt"))

MAX_RETRIES = 3
RETRY_DELAY = 5          # seconds between retries
BATCH_SAVE_INTERVAL = 50  # save progress every N images

# DashScope error codes that indicate quota/balance exhaustion —
# retrying the same model won't help, so we fall back to the next model.
_QUOTA_ERROR_CODES = {
    "Arrearage", "Throttling.RateQuota", "Throttling.AllocationQuota",
    "QuotaExceeded", "ResourceExhausted",
}

# ──────────────────── Helpers ──────────────────────────

def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_vlm(image_path: str, prompt_text: str,
             cat_name: str = "", dtype_name: str = "") -> str:
    """Call Qwen-VL via DashScope native SDK and return the description string.

    Automatically falls back to cheaper models (qwen-vl-plus → qwen-vl-turbo)
    when the current model hits quota or balance errors.

    Parameters
    ----------
    cat_name : str
        MVTec product category (e.g. 'bottle', 'hazelnut').
    dtype_name : str
        MVTec defect type (e.g. 'broken_large', 'scratch').
    """
    b64 = encode_image_base64(image_path)

    # Build user message with image + metadata so the VLM sees the labels
    user_content = [{"image": f"data:image/png;base64,{b64}"}]
    if cat_name or dtype_name:
        meta_parts = []
        if cat_name:
            meta_parts.append(f"product_category: {cat_name}")
        if dtype_name:
            meta_parts.append(f"defect_type: {dtype_name}")
        meta_text = "Known metadata (use these exact labels when consistent with the image): " + "; ".join(meta_parts)
        user_content.append({"text": meta_text})

    last_error = None
    for model_idx, model_name in enumerate(MODEL_FALLBACK):
        for attempt in range(MAX_RETRIES):
            try:
                resp = MultiModalConversation.call(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": [{"text": prompt_text}]},
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                )
                if resp.status_code == 200:
                    return resp.output.choices[0].message.content[0]["text"].strip()
                else:
                    raise RuntimeError(
                        f"DashScope error {resp.status_code}: {resp.code} - {resp.message}"
                    )
            except Exception as e:
                last_error = e
                # Extract error code for quota detection
                code = ""
                try:
                    code = resp.code
                except (NameError, AttributeError):
                    pass
                # Quota exhausted → don't retry this model, fall back
                if code in _QUOTA_ERROR_CODES:
                    tqdm.write(f"  [quota] {model_name} exhausted, falling back...")
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(
        f"Failed after all models {MODEL_FALLBACK}: {last_error}"
    ) from last_error


# ──────────────────── Resume support ──────────────────

def load_progress(output_file: str):
    """Load previously saved JSON so we can skip already-described images."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_progress(data: dict, output_file: str):
    """Atomically save JSON to disk."""
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_file)


# ──────────────────── Main ────────────────────────────

def main():
    # Read API key from environment variable
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: Please set DASHSCOPE_API_KEY environment variable first!")
        print("  CMD:       set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx")
        print("  PowerShell: $env:DASHSCOPE_API_KEY='sk-xxxxxxxxxxxx'")
        sys.exit(1)
    dashscope.api_key = api_key
    print(f"API Key loaded (sk-...{api_key[-6:]})")

    prompt_text = load_prompt(PROMPT_FILE)

    # Resume or initialise
    coco = load_progress(OUTPUT_FILE)
    if coco is not None:
        print(f"[resume] Loaded existing JSON with {len(coco['images'])} images.")
    else:
        coco = {
            "info": {
                "description": "MVTec AD defect image descriptions generated by Qwen-VL",
                "version": "1.0",
                "year": 2026,
                "date_created": datetime.now().isoformat(),
            },
            "images": [],
            "annotations": [],
            "categories": [],
            "defect_types": [],
        }

    # Build category / defect-type lookup
    cat_id_map = {}
    dtype_id_map = {}
    next_cat_id = 1
    next_dtype_id = 1

    categories_list = sorted(os.listdir(MVTEC_ROOT))
    for cat_name in categories_list:
        cat_dir = os.path.join(MVTEC_ROOT, cat_name, "test")
        if not os.path.isdir(cat_dir):
            continue
        cat_id = next_cat_id
        next_cat_id += 1
        cat_id_map[cat_name] = cat_id
        coco["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "industrial_product",
        })
        for dtype_name in sorted(os.listdir(cat_dir)):
            if dtype_name == "good":
                continue
            dtype_dir = os.path.join(cat_dir, dtype_name)
            if not os.path.isdir(dtype_dir):
                continue
            key = f"{cat_name}/{dtype_name}"
            if key not in dtype_id_map:
                dtype_id_map[key] = next_dtype_id
                next_dtype_id += 1
                coco["defect_types"].append({
                    "id": dtype_id_map[key],
                    "name": dtype_name,
                    "category_id": cat_id,
                    "category_name": cat_name,
                })

    # Collect all defect image paths
    all_images = []
    for cat_name in categories_list:
        test_dir = os.path.join(MVTEC_ROOT, cat_name, "test")
        if not os.path.isdir(test_dir):
            continue
        for dtype_name in sorted(os.listdir(test_dir)):
            if dtype_name == "good":
                continue
            dtype_dir = os.path.join(test_dir, dtype_name)
            if not os.path.isdir(dtype_dir):
                continue
            for fname in sorted(os.listdir(dtype_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                all_images.append((cat_name, dtype_name, fname))

    # Determine which images are already processed (for resume)
    done_filenames = {img["file_name"] for img in coco["images"]}
    todo = [
        (cat, dtype, fname)
        for cat, dtype, fname in all_images
        if f"{cat}/{dtype}/{fname}" not in done_filenames
    ]

    print(f"Total defect images : {len(all_images)}")
    print(f"Already processed : {len(done_filenames)}")
    print(f"Remaining         : {len(todo)}")
    print(f"Model             : {MODEL}")
    print()

    next_img_id = max([img["id"] for img in coco["images"]], default=0) + 1
    next_ann_id = max([ann["id"] for ann in coco["annotations"]], default=0) + 1

    processed_in_batch = 0
    pbar = tqdm(total=len(todo), desc="Describing images", unit="img")

    for cat_name, dtype_name, fname in todo:
        img_path = os.path.join(MVTEC_ROOT, cat_name, "test", dtype_name, fname)
        rel_path = f"{cat_name}/{dtype_name}/{fname}"
        t0 = time.time()

        try:
            description = call_vlm(img_path, prompt_text,
                                     cat_name=cat_name, dtype_name=dtype_name)
        except Exception as e:
            description = f"[ERROR] {e}"
            tqdm.write(f"  !! FAILED {rel_path}: {e}")

        elapsed = round(time.time() - t0, 2)

        img_id = next_img_id
        ann_id = next_ann_id
        next_img_id += 1
        next_ann_id += 1

        coco["images"].append({
            "id": img_id,
            "file_name": rel_path,
            "category": cat_name,
            "defect_type": dtype_name,
            "width": 256,
            "height": 256,
        })
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id_map[cat_name],
            "defect_type_id": dtype_id_map[f"{cat_name}/{dtype_name}"],
            "description": description,
            "source_model": MODEL,
        })

        processed_in_batch += 1
        pbar.update(1)

        # Periodic save
        if processed_in_batch % BATCH_SAVE_INTERVAL == 0:
            save_progress(coco, OUTPUT_FILE)
            tqdm.write(f"  [saved] {len(coco['images'])}/{len(all_images)} done")

    pbar.close()

    # Final save
    save_progress(coco, OUTPUT_FILE)

    # Summary
    n_success = sum(1 for a in coco["annotations"] if not a["description"].startswith("[ERROR]"))
    n_fail = sum(1 for a in coco["annotations"] if a["description"].startswith("[ERROR]"))
    print(f"\nDone! Total: {len(coco['images'])} | Success: {n_success} | Failed: {n_fail}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
