"""
ago_optimizer.py
=================
Decomposed AGO (Attribute-Guided Optimization) for VLM-described MVTec images.

Uses Score Distillation Sampling (SDS) to optimize text embeddings of
VLM-generated decomposed descriptions (background / defect / position)
against real anomaly images from the MVTec AD dataset.

The optimized embeddings capture dataset-specific visual semantics and
can be used to improve diffusion-based anomaly generation.

Usage::

    from ago_module import DecomposedAGO

    ago = DecomposedAGO(
        model_path="runwayml/stable-diffusion-v1-5",
        device="cuda",
    )
    embeddings = ago.optimize(
        vlm_description_json=desc_dict,   # one VLM output entry
        reference_image_path=".../bottle/test/broken_large/000.png",
        category="bottle",
        defect_type="broken_large",
        stem="000",
    )
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import trange
from typing import Dict, List, Optional, Tuple

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from .embedding_bank import EmbeddingBank

logger = logging.getLogger(__name__)

# ────────────────── CLIP token limit ─────────────────────────
# SD 1.5 uses CLIP ViT-L/14 with max 77 tokens including BOS/EOS.
# Tokens beyond this are silently truncated — we must validate.
CLIP_MAX_TOKENS = 77


# ────────────────── SDS wrapper ──────────────────────────────

class _SDSModel(torch.nn.Module):
    """Lightweight SDS (Score Distillation Sampling) wrapper.

    Adapted from O2MAG's ``prompt_optimize.StableDiffusion`` to support
    per-component embedding optimization with independent gradient flow.
    """

    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = True,
        model_path: str = "runwayml/stable-diffusion-v1-5",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype
        )
        pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        if is_xformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler", torch_dtype=self.dtype
        )
        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = 1
        self.max_step = self.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        # Registry for named embeddings
        self.embeddings: Dict[str, torch.Tensor] = {}

    # ── token counting ────────────────────────────────────────

    def count_tokens(self, prompt: str) -> int:
        """Return the token count of a prompt **without** truncation."""
        encoded = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        return encoded.input_ids.shape[1]

    # ── text encoding ─────────────────────────────────────────

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenize and encode a single prompt → [1, 77, 768].

        Uses explicit ``truncation=True`` so the tokenizer clips to 77.
        The caller is responsible for checking token length beforehand.
        """
        inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(inputs.input_ids.to(self.device))[0]

    def set_embeddings(self, pos_prompts: Dict[str, str],
                       neg_prompt: str = ""):
        """Register named positive embeddings and a shared negative embedding.

        Parameters
        ----------
        pos_prompts : dict
            Mapping of component_name → prompt_string.
        neg_prompt : str
            Shared negative prompt for CFG.
        """
        self.embeddings.clear()
        for name, prompt in pos_prompts.items():
            emb = self.encode_text(prompt).clone().detach()
            self.embeddings[name] = emb
        neg_emb = self.encode_text(neg_prompt) if neg_prompt else self.encode_text("")
        self.embeddings["neg"] = neg_emb.clone().detach()

    # ── SDS training step ─────────────────────────────────────

    def train_step(
        self,
        pred_rgb: torch.Tensor,
        embed_key: str = "positive",
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Single SDS training step for one named embedding.

        Parameters
        ----------
        pred_rgb : [1, 3, H, W] float tensor in [0, 1].
        embed_key : which registered embedding to optimize against.
        guidance_scale : CFG scale.

        Returns
        -------
        SDS loss scalar.
        """
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        # Encode image to latent
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        latents = self._encode_imgs(pred_rgb_512)

        # Timestep
        t = torch.randint(self.min_step, self.max_step, (batch_size,),
                          dtype=torch.long, device=self.device)
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # Add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # CFG: duplicate for cond + uncond
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)

        pos_emb = self.embeddings[embed_key].expand(batch_size, -1, -1)
        neg_emb = self.embeddings["neg"].expand(batch_size, -1, -1)
        emb = torch.cat([neg_emb, pos_emb], dim=0)

        noise_pred = self.unet(
            latent_model_input, tt, encoder_hidden_states=emb
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return loss

    def get_embedding(self, key: str = "positive") -> torch.Tensor:
        """Return a detached clone of the named embedding."""
        return self.embeddings[key].detach().clone()

    def _encode_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] in [0,1] → latents."""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor


# ────────────────── Decomposed AGO ───────────────────────────

class DecomposedAGO:
    """Optimize VLM-decomposed descriptions via SDS.

    Given a VLM output dict containing ``background_prompt``,
    ``defect_prompt``, ``position_prompt``, ``ago_positive_prompt``,
    and ``ago_negative_prompt``, this class optimises each component's
    text embedding independently against a reference anomaly image.

    Parameters
    ----------
    model_path : str
        Path or HF repo id for Stable Diffusion v1.5.
    device : str
        Torch device string (default ``"cuda"``).
    fp16 : bool
        Use fp16 for the UNet/VAE (default True).
    bank_root : str or None
        If provided, cache optimized embeddings to disk.
    """

    COMPONENT_KEYS = [
        "background_prompt",
        "defect_prompt",
        "position_prompt",
        "ago_positive_prompt",
        "ago_negative_prompt",
    ]

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        fp16: bool = True,
        bank_root: Optional[str] = None,
    ):
        self.device = device
        self.model = _SDSModel(device=device, fp16=fp16, model_path=model_path)
        self.bank = EmbeddingBank(bank_root) if bank_root else None

    # ── token validation ──────────────────────────────────────

    def check_tokens(self, prompts: Dict[str, str]) -> Dict[str, int]:
        """Check token counts for all prompts.  Returns ``{key: n_tokens}``.

        Prints a warning for any prompt that exceeds ``CLIP_MAX_TOKENS`` (77).
        """
        counts: Dict[str, int] = {}
        for key, text in prompts.items():
            n = self.model.count_tokens(text)
            counts[key] = n
            if n > CLIP_MAX_TOKENS:
                logger.warning(
                    "[TOKEN OVERFLOW] %s: %d tokens (limit=%d). "
                    "The last %d tokens will be truncated by CLIP — "
                    "consider shortening this prompt.",
                    key, n, CLIP_MAX_TOKENS, n - CLIP_MAX_TOKENS,
                )
        return counts

    # ── main entry point ──────────────────────────────────────

    def optimize(
        self,
        vlm_description: dict,
        reference_image_path: str,
        category: str = "",
        defect_type: str = "",
        stem: str = "",
        steps: int = 500,
        lr: float = 3e-3,
        guidance_scale: float = 7.5,
        force_reoptimize: bool = False,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run SDS optimisation for all description components.

        Parameters
        ----------
        vlm_description : dict
            One entry from the VLM COCO-like JSON, containing at minimum
            ``ago_positive_prompt`` and ``ago_negative_prompt``. Ideally
            also ``background_prompt``, ``defect_prompt``, ``position_prompt``.
        reference_image_path : str
            Path to the real anomaly image. Used as SDS target.
        category : str
            MVTec category name (for bank lookup).
        defect_type : str
            MVTec defect type (for bank lookup).
        stem : str
            Image stem / unique id (for bank lookup).
        steps : int
            Number of SDS optimisation steps per component.
        lr : float
            Learning rate for embedding optimisation.
        guidance_scale : float
            CFG scale for SDS.
        force_reoptimize : bool
            If True, ignore cached embeddings.
        verbose : bool
            Print progress.

        Returns
        -------
        Dict[str, torch.Tensor]
            Mapping of component_key → optimised embedding [1, 77, 768].
        """
        # Bank hit
        if self.bank is not None and not force_reoptimize:
            if self.bank.exists(category, defect_type, stem):
                cached = self.bank.load(category, defect_type, stem)
                if cached is not None:
                    if verbose:
                        print(f"[Bank] HIT  {category}/{defect_type}/{stem}")
                    return cached

        if verbose:
            print(f"[Bank] MISS → optimising  {category}/{defect_type}/{stem}")

        # Load reference image
        ref_image = self._load_image_512(reference_image_path)

        # Collect components present in the VLM output
        pos_prompts: Dict[str, str] = {}
        for key in self.COMPONENT_KEYS:
            val = vlm_description.get(key, "")
            if val and isinstance(val, str) and val.strip():
                pos_prompts[key] = val

        if not pos_prompts:
            raise ValueError("VLM description contains no usable prompt fields.")

        neg_prompt = vlm_description.get("ago_negative_prompt", "")

        # ── validate token lengths before encoding ────────────
        all_prompts = {**pos_prompts, "ago_negative_prompt": neg_prompt}
        token_counts = self.check_tokens(all_prompts)
        if verbose:
            overflow_keys = [k for k, n in token_counts.items() if n > CLIP_MAX_TOKENS]
            if overflow_keys:
                print(f"[!] Token overflow in: {', '.join(overflow_keys)}")
            else:
                max_key = max(token_counts, key=token_counts.get)
                print(f"[OK] All prompts within {CLIP_MAX_TOKENS}-token limit "
                      f"(max: {max_key}={token_counts[max_key]} tokens)")

        # Initialise embeddings (CLIP will truncate to 77 tokens internally)
        self.model.set_embeddings(pos_prompts, neg_prompt)

        # Freeze model weights
        for p in self.model.parameters():
            p.requires_grad_(False)

        optimised: Dict[str, torch.Tensor] = {}

        # Optimise each component independently
        components = list(pos_prompts.keys())
        for comp_key in components:
            self.model.embeddings[comp_key].requires_grad_(True)
            optimizer = Adam([self.model.embeddings[comp_key]], lr=lr)

            pbar = trange(steps, desc=f"  {comp_key}", leave=False) if verbose else range(steps)
            for _ in pbar:
                optimizer.zero_grad()
                loss = self.model.train_step(
                    ref_image.clone().detach(),
                    embed_key=comp_key,
                    guidance_scale=guidance_scale,
                )
                loss.backward(retain_graph=True)
                optimizer.step()
                if verbose and isinstance(pbar, trange):
                    pbar.set_description(f"  {comp_key} loss={loss.item():.4f}")

            optimised[comp_key] = self.model.get_embedding(comp_key).cpu()
            self.model.embeddings[comp_key].requires_grad_(False)

        # Cache
        if self.bank is not None:
            self.bank.save(
                category, defect_type, stem,
                embeddings={k: v.clone() for k, v in optimised.items()},
                meta={"steps": steps, "lr": lr, "guidance_scale": guidance_scale},
            )

        return optimised

    # ── batch optimisation ────────────────────────────────────

    def optimize_batch(
        self,
        descriptions_json_path: str,
        mvtec_root: str,
        steps: int = 500,
        lr: float = 3e-3,
        guidance_scale: float = 7.5,
        force_reoptimize: bool = False,
        verbose: bool = True,
    ) -> None:
        """Run AGO on all images referenced in a VLM descriptions JSON.

        Parameters
        ----------
        descriptions_json_path : str
            Path to the COCO-style JSON produced by ``generate_descriptions.py``.
        mvtec_root : str
            Root directory of the MVTec AD dataset.
        steps, lr, guidance_scale :
            Passed through to :meth:`optimize`.
        force_reoptimize : bool
            If True, re-run even for cached embeddings.
        verbose : bool
            Print overall progress.
        """
        import json

        with open(descriptions_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build lookup: image_id → annotation
        ann_by_image = {a["image_id"]: a for a in coco.get("annotations", [])}

        images = coco.get("images", [])
        for img_info in images:
            img_id = img_info["id"]
            ann = ann_by_image.get(img_id)
            if ann is None:
                continue

            description_text = ann.get("description", "")
            if description_text.startswith("[ERROR]"):
                continue

            # Parse the VLM JSON output
            try:
                vlm_desc = json.loads(description_text)
            except (json.JSONDecodeError, TypeError):
                if verbose:
                    print(f"[Skip] invalid JSON for {img_info['file_name']}")
                continue

            category = img_info.get("category", "")
            defect_type = img_info.get("defect_type", "")
            file_name = img_info["file_name"]  # e.g. "bottle/broken_large/000.png"
            stem = file_name.split("/")[-1].replace(".png", "")
            ref_path = f"{mvtec_root}/{file_name}"

            try:
                self.optimize(
                    vlm_description=vlm_desc,
                    reference_image_path=ref_path,
                    category=category,
                    defect_type=defect_type,
                    stem=stem,
                    steps=steps,
                    lr=lr,
                    guidance_scale=guidance_scale,
                    force_reoptimize=force_reoptimize,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"[Error] {file_name}: {e}")

    # ── helpers ───────────────────────────────────────────────

    def _load_image_512(self, path: str) -> torch.Tensor:
        """Load image, convert to RGB, resize to 512×512, return [1,3,512,512] in [0,1]."""
        img = Image.open(path).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        tensor = pil_to_tensor(img).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def to(self, device: str):
        """Move the SDS model to a different device."""
        self.device = device
        self.model.to(device)
        return self
