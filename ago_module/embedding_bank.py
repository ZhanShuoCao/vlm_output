"""
embedding_bank.py
=================
Persistent embedding cache for AGO-optimized text embeddings.

Stores optimized embeddings keyed by (category, defect_type, image_stem),
supporting fast lookup to avoid re-optimization.
"""

import os
import json
import hashlib
import torch
from pathlib import Path
from typing import Optional, Dict


class EmbeddingBank:
    """Disk-backed cache for optimized prompt embeddings.

    Each entry is stored as a .pt file under:
        {bank_root}/{category}/{defect_type}/{stem}.pt

    A lightweight index file (index.json) tracks metadata.
    """

    def __init__(self, bank_root: str):
        self.root = Path(bank_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index: Dict[str, dict] = self._load_index()

    # ─── public API ───────────────────────────────────────────

    def exists(self, category: str, defect_type: str, stem: str) -> bool:
        """Check whether an optimized embedding already exists."""
        key = self._make_key(category, defect_type, stem)
        return key in self._index and self._pt_path(key).exists()

    def load(self, category: str, defect_type: str, stem: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached embeddings dict. Returns None on miss."""
        key = self._make_key(category, defect_type, stem)
        if key not in self._index:
            return None
        pt = self._pt_path(key)
        if not pt.exists():
            return None
        return torch.load(pt, map_location="cpu", weights_only=True)

    def save(self, category: str, defect_type: str, stem: str,
             embeddings: Dict[str, torch.Tensor],
             meta: Optional[dict] = None):
        """Save a dictionary of embeddings to the bank."""
        key = self._make_key(category, defect_type, stem)
        pt = self._pt_path(key)
        pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, pt)

        entry = {
            "category": category,
            "defect_type": defect_type,
            "stem": stem,
            "components": list(embeddings.keys()),
            "path": str(pt),
        }
        if meta:
            entry["meta"] = meta
        self._index[key] = entry
        self._save_index()

    def remove(self, category: str, defect_type: str, stem: str):
        """Remove a single entry."""
        key = self._make_key(category, defect_type, stem)
        pt = self._pt_path(key)
        if pt.exists():
            pt.unlink()
        self._index.pop(key, None)
        self._save_index()

    def list_entries(self) -> list:
        """Return list of all index entries."""
        return list(self._index.values())

    # ─── internals ────────────────────────────────────────────

    def _make_key(self, category: str, defect_type: str, stem: str) -> str:
        raw = f"{category}/{defect_type}/{stem}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _pt_path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pt"

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
