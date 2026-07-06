"""
generate_descriptions.py
========================
Use Qwen-VL via 阿里云百炼 OpenAI-compatible API to generate per-image visual
descriptions for MVTec AD defect images, output a COCO-style JSON annotation file.

Prerequisites:
    pip install openai tqdm

Usage:
    # Step 1: Set environment variables
    set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    set MVTEC_ROOT=D:\path\to\MVTec
    set OUTPUT_FILE=./mvtec_descriptions.json

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

from openai import OpenAI
from tqdm import tqdm

# ──────────────────── Configuration ────────────────────
# 百炼 OpenAI-compatible endpoint
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Model fallback chain — uses 百炼 models covered by free quota
MODEL_FALLBACK = [
    "qwen-vl-max",      # Qwen2.5-VL-72B, best quality
    "qwen-vl-plus",     # Qwen2.5-VL-7B, lighter/cheaper
]

MVTEC_ROOT = os.environ.get("MVTEC_ROOT", r"D:\Desktop\MVTec")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "./mvtec_descriptions.json")
PROMPT_FILE = os.environ.get(
    "PROMPT_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_prompt_mvtec_ago_template.txt"),
)

MAX_RETRIES = 3
RETRY_DELAY = 5          # seconds between retries
BATCH_SAVE_INTERVAL = 50  # save progress every N images

# ──────────────────── OpenAI client ────────────────────

def _build_client() -> OpenAI:
    """Build an OpenAI client pointed at 百炼, with proxy explicitly disabled."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    import httpx

    # Explicitly create an httpx client that does NOT use system proxy
    # (SOCKS proxy often causes Connection refused errors on 百炼)
    http_client = httpx.Client(
        proxy=None,       # disable system proxy
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        http_client=http_client,
    )

# ──────────────────── Helpers ──────────────────────────

def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_vlm(image_path: str, prompt_text: str,
             cat_name: str = "", dtype_name: str = "") -> str:
    """Call Qwen-VL via 百炼 OpenAI-compatible API and return the description string.

    Automatically falls back to cheaper models when the current model hits
    quota or balance errors.
    """
    b64 = encode_image_base64(image_path)

    # Build user message content with image + metadata
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
    ]
    if cat_name or dtype_name:
        meta_parts = []
        if cat_name:
            meta_parts.append(f"product_category: {cat_name}")
        if dtype_name:
            meta_parts.append(f"defect_type: {dtype_name}")
        meta_text = (
            "Known metadata (use these exact labels when consistent with the image): "
            + "; ".join(meta_parts)
        )
        user_content.append({"type": "text", "text": meta_text})

    client = _build_client()
    last_error = None

    for model_name in MODEL_FALLBACK:
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                msg = str(e)

                # Detect quota/balance errors → don't retry, fall back
                if _is_quota_error(msg):
                    tqdm.write(f"  [quota] {model_name} exhausted, falling back...")
                    break

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (attempt + 1)
                    tqdm.write(f"  [retry] {model_name} attempt {attempt+1}/{MAX_RETRIES}, "
                               f"waiting {delay}s: {msg[:120]}")
                    time.sleep(delay)

    raise RuntimeError(
        f"Failed after all models {MODEL_FALLBACK}: {last_error}"
    ) from last_error


def _is_quota_error(msg: str) -> bool:
    """Detect quota/balance exhaustion from error message text."""
    quota_keywords = [
        "Arrearage", "Throttling", "QuotaExceeded", "ResourceExhausted",
        "quota", "insufficient", "balance", "arrears", "AccountArrears",
        "out of quota", "rate limit", "limit exceeded",
    ]
    msg_lower = msg.lower()
    return any(kw.lower() in msg_lower for kw in quota_keywords)


# ──────────────────── Resume support ──────────────────

def load_progress(output_file: str):
    """Load previously saved JSON so we can skip already-described images."""
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Filter out error entries so they get retried
            good = [a for a in data.get("annotations", [])
                    if not a["description"].startswith("[ERROR]")]
            good_filenames = {
                img["file_name"]
                for img in data.get("images", [])
                if img["id"] in {a["image_id"] for a in good}
            }
            # Remove failed entries so they'll be reprocessed
            data["images"] = [img for img in data["images"]
                              if img["file_name"] in good_filenames]
            data["annotations"] = good
            return data
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
    print(f"API Key loaded (sk-...{api_key[-6:]})")
    print(f"Base URL: {BASE_URL}")
    print(f"MVTec root: {MVTEC_ROOT}")

    prompt_text = load_prompt(PROMPT_FILE)

    # Resume or initialise
    coco = load_progress(OUTPUT_FILE)
    if coco is not None:
        print(f"[resume] Loaded existing JSON with {len(coco['images'])} valid images "
              f"(failed entries discarded for retry).")
    else:
        coco = {
            "info": {
                "description": "MVTec AD defect image descriptions generated by Qwen-VL (百炼 OpenAI API)",
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
        coco.setdefault("categories", []).append({
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
                coco.setdefault("defect_types", []).append({
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

    # Determine which images need processing
    done_filenames = {img["file_name"] for img in coco.get("images", [])}
    todo = [
        (cat, dtype, fname)
        for cat, dtype, fname in all_images
        if f"{cat}/{dtype}/{fname}" not in done_filenames
    ]

    print(f"Total defect images : {len(all_images)}")
    print(f"Already processed   : {len(done_filenames)}")
    print(f"Remaining           : {len(todo)}")
    print(f"Model fallback chain: {MODEL_FALLBACK}")
    print()

    next_img_id = max([img["id"] for img in coco.get("images", [])], default=0) + 1
    next_ann_id = max([ann["id"] for ann in coco.get("annotations", [])], default=0) + 1

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
            "source_model": "openai-compat",
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
