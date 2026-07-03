"""
AGO (Attribute-Guided Optimization) Module
===========================================
VLM-enhanced semantic prompt optimization for MVTec anomaly generation.

This module takes VLM-generated decomposed descriptions (background, defect,
position) and uses Score Distillation Sampling (SDS) to optimize their text
embeddings against real anomaly images. The optimized embeddings can then be
used in diffusion-based anomaly generation pipelines.

Core components:
- DecomposedAGO: main optimizer for per-component embedding optimization
- EmbeddingBank: persistent cache for optimized embeddings
"""

from .ago_optimizer import DecomposedAGO
from .embedding_bank import EmbeddingBank

__all__ = ["DecomposedAGO", "EmbeddingBank"]
