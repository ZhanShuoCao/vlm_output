#!/usr/bin/env python3
"""
run_triple_ago.py
==================
CLI for three-pipeline AGO (background / defect / position).

Usage::

    python run_triple_ago.py \
        --descriptions ./datasets/mvtec_descriptions.json \
        --mvtec-root D:\\Novels_Code\\open_datasets\\MVTec \
        --bank-root ./triple_embed_bank \
        --bg-steps 500 --def-steps 500 --pos-steps 300 \
        --use-mask \
        --device cuda

Output:
    Three independent optimised embeddings per image, cached under
    ``--bank-root``. Each entry contains keys ``background``, ``defect``,
    ``position``.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ago_decomposed can import ago_module
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    ap = argparse.ArgumentParser(
        description="Triple-pipeline AGO: independent background / defect / position SDS optimisation"
    )
    ap.add_argument("--descriptions", required=True,
                    help="COCO JSON from generate_descriptions.py")
    ap.add_argument("--mvtec-root", required=True,
                    help="MVTec AD dataset root")
    ap.add_argument("--model-path", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--bank-root", default="./triple_embed_bank")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true",
                    help="Use fp32 (default fp16)")

    # Per-pipeline steps
    ap.add_argument("--bg-steps", type=int, default=500,
                    help="SDS steps for background pipeline")
    ap.add_argument("--def-steps", type=int, default=500,
                    help="SDS steps for defect pipeline")
    ap.add_argument("--pos-steps", type=int, default=300,
                    help="SDS steps for position pipeline")

    # Per-pipeline LR
    ap.add_argument("--bg-lr", type=float, default=3e-3)
    ap.add_argument("--def-lr", type=float, default=3e-3)
    ap.add_argument("--pos-lr", type=float, default=3e-3)

    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--use-mask", action="store_true",
                    help="Enable masked SDS (requires MVTec ground_truth masks)")
    ap.add_argument("--force", action="store_true",
                    help="Re-optimise cached entries")
    ap.add_argument("--category", default="",
                    help="Filter to single category")
    ap.add_argument("--defect-type", default="",
                    help="Filter to single defect type")
    args = ap.parse_args()

    # ── validate ──────────────────────────────────────────────
    if not os.path.isfile(args.descriptions):
        print(f"ERROR: descriptions not found: {args.descriptions}")
        sys.exit(1)
    if not os.path.isdir(args.mvtec_root):
        print(f"ERROR: MVTec root not found: {args.mvtec_root}")
        sys.exit(1)

    # ── imports (after args parsed) ───────────────────────────
    from ago_decomposed.triple_ago import TripleAGO, PipelineConfig

    bg_cfg = PipelineConfig(
        steps=args.bg_steps, lr=args.bg_lr,
        guidance_scale=args.guidance_scale, use_mask=args.use_mask,
        mask_region="background",
    )
    def_cfg = PipelineConfig(
        steps=args.def_steps, lr=args.def_lr,
        guidance_scale=args.guidance_scale, use_mask=args.use_mask,
        mask_region="defect",
    )
    pos_cfg = PipelineConfig(
        steps=args.pos_steps, lr=args.pos_lr,
        guidance_scale=args.guidance_scale, use_mask=False,
    )

    print("=" * 60)
    print("Triple-Pipeline AGO")
    print("=" * 60)
    print(f"  Background : {bg_cfg.steps} steps, lr={bg_cfg.lr}, masked={bg_cfg.use_mask}")
    print(f"  Defect     : {def_cfg.steps} steps, lr={def_cfg.lr}, masked={def_cfg.use_mask}")
    print(f"  Position   : {pos_cfg.steps} steps, lr={pos_cfg.lr}, masked={pos_cfg.use_mask}")
    print(f"  Guidance   : {args.guidance_scale}")
    print(f"  Bank       : {args.bank_root}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    triple = TripleAGO(
        model_path=args.model_path,
        device=args.device,
        fp16=not args.fp32,
        bank_root=args.bank_root,
        bg_config=bg_cfg,
        def_config=def_cfg,
        pos_config=pos_cfg,
    )

    # ── load and filter ───────────────────────────────────────
    import json
    from tqdm import tqdm

    with open(args.descriptions, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_by_image = {a["image_id"]: a for a in coco.get("annotations", [])}
    images = coco.get("images", [])
    if args.category:
        images = [i for i in images if i.get("category") == args.category]
    if args.defect_type:
        images = [i for i in images if i.get("defect_type") == args.defect_type]

    n_ok = n_skip = n_err = 0
    pbar = tqdm(images, desc="Triple AGO", unit="img")

    for img_info in pbar:
        ann = ann_by_image.get(img_info["id"])
        if ann is None:
            n_skip += 1
            continue

        desc_text = ann.get("description", "")
        if desc_text.startswith("[ERROR]"):
            n_skip += 1
            continue

        try:
            vlm_desc = json.loads(desc_text)
        except (json.JSONDecodeError, TypeError):
            tqdm.write(f"  [Skip] bad JSON: {img_info['file_name']}")
            n_skip += 1
            continue

        category = img_info.get("category", "")
        defect_type = img_info.get("defect_type", "")
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        ref_path = os.path.join(args.mvtec_root, file_name)

        mask_path = None
        if args.use_mask:
            gt_mask = Path(args.mvtec_root) / category / "ground_truth" / defect_type / f"{stem}_mask.png"
            if gt_mask.is_file():
                mask_path = str(gt_mask)

        try:
            triple.optimize(
                vlm_description=vlm_desc,
                reference_image_path=ref_path,
                mask_path=mask_path,
                category=category,
                defect_type=defect_type,
                stem=stem,
                force_reoptimize=args.force,
                verbose=False,
            )
            n_ok += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
        except Exception as e:
            tqdm.write(f"  [Error] {file_name}: {e}")
            n_err += 1

    print(f"\nDone!  OK: {n_ok}  |  Skipped: {n_skip}  |  Errors: {n_err}")
    print(f"Bank: {Path(args.bank_root).resolve()}")
    if triple.bank:
        print(f"Entries: {len(triple.bank.list_entries())}")


if __name__ == "__main__":
    main()
