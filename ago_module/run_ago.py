#!/usr/bin/env python3
"""
run_ago.py
==========
CLI entry point for AGO (Attribute-Guided Optimization) on MVTec AD.

Loads VLM-generated descriptions, then runs per-component SDS optimisation
to produce refined text embeddings for anomaly generation.

Usage::

    # Step 1: set HF token if model is gated
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

    # Step 2: run optimisation
    python run_ago.py \\
        --descriptions ./datasets/mvtec_descriptions.json \\
        --mvtec-root D:\\Novels_Code\\open_datasets\\MVTec \\
        --bank-root ./mvtec_embed_bank \\
        --steps 500 \\
        --device cuda

Output:
    Optimised embeddings are saved to ``--bank-root``, organised as
    ``{bank_root}/{hash_prefix}/{hash}.pt`` with an ``index.json`` manifest.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="AGO: VLM-enhanced prompt embedding optimisation for MVTec AD"
    )
    ap.add_argument(
        "--descriptions", required=True,
        help="Path to COCO-style JSON from generate_descriptions.py",
    )
    ap.add_argument(
        "--mvtec-root", required=True,
        help="Root directory of the MVTec AD dataset",
    )
    ap.add_argument(
        "--model-path", default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion v1.5 path or HF repo id",
    )
    ap.add_argument(
        "--bank-root", default="./ago_embedding_bank",
        help="Directory to cache optimised embeddings",
    )
    ap.add_argument(
        "--steps", type=int, default=500,
        help="SDS optimisation steps per component (default: 500)",
    )
    ap.add_argument(
        "--lr", type=float, default=3e-3,
        help="Learning rate for embedding optimisation",
    )
    ap.add_argument(
        "--guidance-scale", type=float, default=7.5,
        help="CFG scale for SDS",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Torch device, e.g. 'cuda:0' or 'cpu'",
    )
    ap.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use fp16 for model weights (default: True)",
    )
    ap.add_argument(
        "--fp32", action="store_true",
        help="Use fp32 instead of fp16",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-optimise even if cached embeddings exist",
    )
    ap.add_argument(
        "--category", default="",
        help="Only process this category (e.g. 'bottle'). Empty = all.",
    )
    ap.add_argument(
        "--defect-type", default="",
        help="Only process this defect type. Empty = all.",
    )
    args = ap.parse_args()

    # ── validate inputs ───────────────────────────────────────
    if not os.path.isfile(args.descriptions):
        print(f"ERROR: descriptions file not found: {args.descriptions}")
        sys.exit(1)
    if not os.path.isdir(args.mvtec_root):
        print(f"ERROR: MVTec root not found: {args.mvtec_root}")
        sys.exit(1)

    fp16 = args.fp16 and not args.fp32

    print("=" * 60)
    print("AGO  —  Attribute-Guided Optimisation")
    print("=" * 60)
    print(f"  Descriptions : {args.descriptions}")
    print(f"  MVTec root   : {args.mvtec_root}")
    print(f"  Model        : {args.model_path}")
    print(f"  Bank         : {args.bank_root}")
    print(f"  Steps        : {args.steps}")
    print(f"  LR           : {args.lr}")
    print(f"  Guidance     : {args.guidance_scale}")
    print(f"  Device       : {args.device}")
    print(f"  Precision    : {'fp16' if fp16 else 'fp32'}")
    if args.category:
        print(f"  Filter cat   : {args.category}")
    if args.defect_type:
        print(f"  Filter dtype : {args.defect_type}")
    print("=" * 60)

    # ── import after args parsed (avoids slow import on --help) ─
    from ago_module import DecomposedAGO
    import json
    from tqdm import tqdm

    ago = DecomposedAGO(
        model_path=args.model_path,
        device=args.device,
        fp16=fp16,
        bank_root=args.bank_root,
    )

    # ── load descriptions ─────────────────────────────────────
    with open(args.descriptions, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_by_image = {a["image_id"]: a for a in coco.get("annotations", [])}
    images = coco.get("images", [])

    # Apply filters
    if args.category:
        images = [i for i in images if i.get("category") == args.category]
    if args.defect_type:
        images = [i for i in images if i.get("defect_type") == args.defect_type]

    n_total = len(images)
    n_skipped = 0
    n_errors = 0
    n_optimised = 0

    pbar = tqdm(images, desc="AGO optimisation", unit="img")
    for img_info in pbar:
        img_id = img_info["id"]
        ann = ann_by_image.get(img_id)
        if ann is None:
            n_skipped += 1
            continue

        description_text = ann.get("description", "")
        if description_text.startswith("[ERROR]"):
            if args.force:
                tqdm.write(f"  [Skip] ERROR description for {img_info['file_name']}")
            n_skipped += 1
            continue

        # Parse the VLM JSON
        try:
            vlm_desc = json.loads(description_text)
        except (json.JSONDecodeError, TypeError):
            tqdm.write(f"  [Skip] invalid JSON: {img_info['file_name']}")
            n_skipped += 1
            continue

        category = img_info.get("category", "")
        defect_type = img_info.get("defect_type", "")
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        ref_path = os.path.join(args.mvtec_root, file_name)

        if not os.path.isfile(ref_path):
            tqdm.write(f"  [Skip] missing image: {ref_path}")
            n_skipped += 1
            continue

        try:
            ago.optimize(
                vlm_description=vlm_desc,
                reference_image_path=ref_path,
                category=category,
                defect_type=defect_type,
                stem=stem,
                steps=args.steps,
                lr=args.lr,
                guidance_scale=args.guidance_scale,
                force_reoptimize=args.force,
                verbose=False,  # per-image progress handled by pbar
            )
            n_optimised += 1
            pbar.set_postfix(optimised=n_optimised, skipped=n_skipped, errors=n_errors)
        except Exception as e:
            tqdm.write(f"  [Error] {file_name}: {e}")
            n_errors += 1

    # ── summary ───────────────────────────────────────────────
    print()
    print(f"Done!  Optimised: {n_optimised}  |  Skipped: {n_skipped}  |  Errors: {n_errors}")
    print(f"Bank: {Path(args.bank_root).resolve()}")
    print(f"Entries in bank: {len(ago.bank.list_entries()) if ago.bank else 0}")


if __name__ == "__main__":
    main()
