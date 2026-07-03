"""
triple_ago.py
==============
Three independent AGO pipelines for background / defect / position.

Each pipeline is a ``SinglePipelineAGO`` that runs SDS optimisation for
exactly one semantic component.  ``TripleAGO`` orchestrates all three,
optionally using MVTec ground-truth masks to focus each pipeline's SDS
loss on the relevant image region.

Architecture::

    VLM description
    ├── background_prompt  ──→  Pipeline-BG  ──→  emb_bg  [1,77,768]
    ├── defect_prompt      ──→  Pipeline-DEF ──→  emb_def [1,77,768]
    └── position_prompt    ──→  Pipeline-POS ──→  emb_pos [1,77,768]

Usage::

    from ago_decomposed import TripleAGO

    triple = TripleAGO(model_path="runwayml/stable-diffusion-v1-5")
    result = triple.optimize(
        vlm_description=vlm_dict,
        reference_image_path=".../000.png",
        mask_path=".../000_mask.png",        # optional ground-truth mask
        category="bottle",
        defect_type="broken_large",
        stem="000",
    )
    # result["background"]  → optimised background embedding
    # result["defect"]      → optimised defect embedding
    # result["position"]    → optimised position embedding
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import trange
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Reuse the SDS model from ago_module
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ago_module.ago_optimizer import _SDSModel, CLIP_MAX_TOKENS
from ago_module.embedding_bank import EmbeddingBank


# ────────────────── Pipeline config ──────────────────────────

@dataclass
class PipelineConfig:
    """Hyperparameters for one AGO pipeline."""
    steps: int = 500
    lr: float = 3e-3
    guidance_scale: float = 7.5
    # If True, apply a spatial mask to the SDS loss so only the
    # relevant image region contributes. Requires a ground-truth mask.
    use_mask: bool = False
    # "defect" = optimise against the defect region (mask=1 area).
    # "background" = optimise against the non-defect region (mask=0 area).
    mask_region: str = "defect"


# ────────────────── Single-pipeline AGO ──────────────────────

class SinglePipelineAGO:
    """SDS optimiser for exactly **one** semantic component.

    Parameters
    ----------
    component_name : str
        Human-readable label (``"background"``, ``"defect"``, ``"position"``).
    model : _SDSModel
        Shared or dedicated SDS wrapper.
    config : PipelineConfig
        Optimisation hyperparameters.
    """

    def __init__(
        self,
        component_name: str,
        model: _SDSModel,
        config: Optional[PipelineConfig] = None,
    ):
        self.name = component_name
        self.model = model
        self.config = config or PipelineConfig()

    # ── optimisation ──────────────────────────────────────────

    def optimize(
        self,
        prompt: str,
        ref_image_512: torch.Tensor,          # [1,3,512,512] in [0,1]
        mask_512: Optional[torch.Tensor] = None,  # [1,1,512,512] or [512,512] in [0,1]
        negative_prompt: str = "",
        verbose: bool = True,
    ) -> torch.Tensor:
        """Run SDS for this single component.

        Parameters
        ----------
        prompt : str
            The text prompt for this component (e.g. ``defect_prompt``).
        ref_image_512 : [1,3,512,512] float in [0,1].
        mask_512 : optional binary mask, same spatial size.
        negative_prompt : CFG negative prompt for this pipeline.
        verbose : bool

        Returns
        -------
        optimised embedding  [1, 77, 768] on CPU.
        """
        # ── token check ───────────────────────────────────────
        n_tok = self.model.count_tokens(prompt)
        if n_tok > CLIP_MAX_TOKENS:
            print(f"[!] {self.name}: {n_tok} tokens (limit={CLIP_MAX_TOKENS}) — truncating")

        # ── register embeddings ───────────────────────────────
        self.model.set_embeddings({self.name: prompt}, negative_prompt or "")

        # Prepare masked reference if requested
        ref_for_loss = ref_image_512
        if self.config.use_mask and mask_512 is not None:
            ref_for_loss = self._apply_mask(ref_image_512, mask_512)

        # ── freeze model, unfreeze this embedding ─────────────
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.embeddings[self.name].requires_grad_(True)

        optimizer = Adam([self.model.embeddings[self.name]], lr=self.config.lr)

        pbar = trange(self.config.steps, desc=f"  [{self.name}]", leave=False) if verbose else range(self.config.steps)
        for _ in pbar:
            optimizer.zero_grad()
            loss = self.model.train_step(
                ref_for_loss.clone().detach(),
                embed_key=self.name,
                guidance_scale=self.config.guidance_scale,
            )
            loss.backward(retain_graph=True)
            optimizer.step()
            if verbose and isinstance(pbar, trange):
                pbar.set_description(f"  [{self.name}] loss={loss.item():.4f}")

        result = self.model.get_embedding(self.name).cpu()
        self.model.embeddings[self.name].requires_grad_(False)
        return result

    # ── mask helpers ──────────────────────────────────────────

    def _apply_mask(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Blend image with its mean colour inside/outside the mask region.

        This forces the SDS loss to focus on the selected region while
        the other region becomes a neutral average — gradient flows
        primarily through the masked area.
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        # mask shape now [1,1,H,W]

        mask = F.interpolate(mask.float(), size=image.shape[-2:], mode="nearest")

        if self.config.mask_region == "defect":
            # Keep defect region visible, replace background with mean
            bg_mask = (mask < 0.5).float()
            mean_color = (image * bg_mask).sum(dim=[-2, -1], keepdim=True) / bg_mask.sum().clamp(min=1)
            image_masked = image * mask + mean_color * (1 - mask)
        else:
            # Keep background visible, replace defect with mean
            fg_mask = (mask > 0.5).float()
            mean_color = (image * (1 - fg_mask)).sum(dim=[-2, -1], keepdim=True) / (1 - fg_mask).sum().clamp(min=1)
            image_masked = image * (1 - mask) + mean_color * mask

        return image_masked


# ────────────────── Triple-pipeline orchestrator ─────────────

class TripleAGO:
    """Orchestrate three independent AGO pipelines.

    Parameters
    ----------
    model_path : str
        Stable Diffusion v1.5 path.
    device : str
    fp16 : bool
    bank_root : str or None
        Cache directory for optimised embeddings.
    bg_config : PipelineConfig
    def_config : PipelineConfig
    pos_config : PipelineConfig
    """

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        fp16: bool = True,
        bank_root: Optional[str] = None,
        bg_config: Optional[PipelineConfig] = None,
        def_config: Optional[PipelineConfig] = None,
        pos_config: Optional[PipelineConfig] = None,
    ):
        self.device = device

        # Each pipeline gets its own SDS model instance so they don't
        # interfere — embeddings are registered independently.
        self.bg_model = _SDSModel(device=device, fp16=fp16, model_path=model_path)
        self.def_model = _SDSModel(device=device, fp16=fp16, model_path=model_path)
        self.pos_model = _SDSModel(device=device, fp16=fp16, model_path=model_path)

        self.pipeline_bg = SinglePipelineAGO("background", self.bg_model, bg_config)
        self.pipeline_def = SinglePipelineAGO("defect", self.def_model, def_config)
        self.pipeline_pos = SinglePipelineAGO("position", self.pos_model, pos_config)

        self.bank = EmbeddingBank(bank_root) if bank_root else None

    # ── main entry point ──────────────────────────────────────

    def optimize(
        self,
        vlm_description: dict,
        reference_image_path: str,
        mask_path: Optional[str] = None,
        category: str = "",
        defect_type: str = "",
        stem: str = "",
        force_reoptimize: bool = False,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run all three pipelines.

        Parameters
        ----------
        vlm_description : dict
            VLM JSON with ``background_prompt``, ``defect_prompt``,
            ``position_prompt``, ``ago_negative_prompt``.
        reference_image_path : str
        mask_path : str or None
            Ground-truth mask for masked SDS (MVTec ground_truth).
        category, defect_type, stem : str
            For bank lookup.
        force_reoptimize : bool
        verbose : bool

        Returns
        -------
        Dict with keys ``"background"``, ``"defect"``, ``"position"``
        each mapping to a ``[1, 77, 768]`` optimised embedding.
        """
        # ── bank hit ──────────────────────────────────────────
        if self.bank is not None and not force_reoptimize:
            if self.bank.exists(category, defect_type, stem):
                cached = self.bank.load(category, defect_type, stem)
                if cached is not None:
                    if verbose:
                        print(f"[Bank] HIT  {category}/{defect_type}/{stem}")
                    return cached

        if verbose:
            print(f"[Bank] MISS → 3-pipeline AGO  {category}/{defect_type}/{stem}")

        # ── load data ─────────────────────────────────────────
        ref_image = self._load_image_512(reference_image_path)
        mask = self._load_mask_512(mask_path) if mask_path else None

        # ── extract prompts ───────────────────────────────────
        bg_prompt = vlm_description.get("background_prompt", "")
        def_prompt = vlm_description.get("defect_prompt", "")
        pos_prompt = vlm_description.get("position_prompt", "")
        neg_prompt = vlm_description.get("ago_negative_prompt", "")

        if not bg_prompt or not def_prompt:
            raise ValueError(
                "VLM description missing background_prompt or defect_prompt. "
                "Both are required for triple-pipeline AGO."
            )

        # ── per-pipeline negative prompts ─────────────────────
        # Derive focused negative prompts so each pipeline pushes
        # away from the *other* semantic dimensions.
        neg_bg = self._make_bg_negative(defect_type, neg_prompt)
        neg_def = self._make_def_negative(category, neg_prompt)
        neg_pos = neg_prompt  # position pipeline uses the shared negative

        # ── run three pipelines ───────────────────────────────
        if verbose:
            print(f"  Pipeline 1/3: background  ({self.pipeline_bg.config.steps} steps)")

        if self.pipeline_bg.config.use_mask and mask is not None:
            # Background pipeline: optimise against non-defect regions
            self.pipeline_bg.config.mask_region = "background"
            emb_bg = self.pipeline_bg.optimize(
                bg_prompt, ref_image, mask,
                negative_prompt=neg_bg, verbose=verbose,
            )
        else:
            emb_bg = self.pipeline_bg.optimize(
                bg_prompt, ref_image,
                negative_prompt=neg_bg, verbose=verbose,
            )

        if verbose:
            print(f"  Pipeline 2/3: defect     ({self.pipeline_def.config.steps} steps)")

        if self.pipeline_def.config.use_mask and mask is not None:
            self.pipeline_def.config.mask_region = "defect"
            emb_def = self.pipeline_def.optimize(
                def_prompt, ref_image, mask,
                negative_prompt=neg_def, verbose=verbose,
            )
        else:
            emb_def = self.pipeline_def.optimize(
                def_prompt, ref_image,
                negative_prompt=neg_def, verbose=verbose,
            )

        if verbose:
            print(f"  Pipeline 3/3: position   ({self.pipeline_pos.config.steps} steps)")

        emb_pos = self.pipeline_pos.optimize(
            pos_prompt, ref_image,
            negative_prompt=neg_pos, verbose=verbose,
        )

        result = {
            "background": emb_bg,
            "defect": emb_def,
            "position": emb_pos,
        }

        # ── cache ─────────────────────────────────────────────
        if self.bank is not None:
            self.bank.save(
                category, defect_type, stem,
                embeddings={k: v.clone() for k, v in result.items()},
                meta={"type": "triple_ago"},
            )

        return result

    # ── negative prompt builders ──────────────────────────────

    @staticmethod
    def _make_bg_negative(defect_type: str, fallback: str) -> str:
        """Negative prompt that pushes background embedding away from defect features."""
        defect_terms = defect_type.replace("_", " ")
        base = (
            f"defect, anomaly, damage, {defect_terms}, scratch, crack, hole, "
            "broken, contamination, stain, discoloration, missing part, "
            "deformation, irregular texture, foreign material"
        )
        if fallback:
            return f"{base}, {fallback}"
        return base

    @staticmethod
    def _make_def_negative(category: str, fallback: str) -> str:
        """Negative prompt that pushes defect embedding away from normal product features."""
        base = (
            f"defect-free, intact, pristine, flawless, clean surface, "
            f"normal {category}, perfect condition, no anomaly, "
            "regular texture, uniform surface"
        )
        if fallback:
            return f"{base}, {fallback}"
        return base

    # ── batch ─────────────────────────────────────────────────

    def optimize_batch(
        self,
        descriptions_json_path: str,
        mvtec_root: str,
        force_reoptimize: bool = False,
        verbose: bool = True,
    ) -> None:
        """Run triple-pipeline AGO on a full VLM descriptions JSON."""
        import json

        with open(descriptions_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        ann_by_image = {a["image_id"]: a for a in coco.get("annotations", [])}

        for img_info in coco.get("images", []):
            ann = ann_by_image.get(img_info["id"])
            if ann is None:
                continue

            desc_text = ann.get("description", "")
            if desc_text.startswith("[ERROR]"):
                continue

            try:
                vlm_desc = json.loads(desc_text)
            except (json.JSONDecodeError, TypeError):
                continue

            category = img_info.get("category", "")
            defect_type = img_info.get("defect_type", "")
            file_name = img_info["file_name"]
            stem = Path(file_name).stem
            ref_path = f"{mvtec_root}/{file_name}"

            # Derive mask path from MVTec convention
            mask_path = None
            gt_mask = Path(mvtec_root) / category / "ground_truth" / defect_type / f"{stem}_mask.png"
            if gt_mask.is_file():
                mask_path = str(gt_mask)

            try:
                self.optimize(
                    vlm_description=vlm_desc,
                    reference_image_path=ref_path,
                    mask_path=mask_path,
                    category=category,
                    defect_type=defect_type,
                    stem=stem,
                    force_reoptimize=force_reoptimize,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[Error] {file_name}: {e}")

    # ── helpers ───────────────────────────────────────────────

    def _load_image_512(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        tensor = pil_to_tensor(img).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _load_mask_512(self, path: str) -> torch.Tensor:
        """Load a binary mask, resize to 512×512, return [512,512] in {0,1}."""
        mask = Image.open(path).convert("L")
        mask = mask.resize((512, 512), Image.NEAREST)
        tensor = pil_to_tensor(mask).float() / 255.0
        tensor = (tensor > 0.5).float()  # binarise
        return tensor.squeeze(0).to(self.device)
