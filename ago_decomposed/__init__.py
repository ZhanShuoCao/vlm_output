"""
AGO Decomposed  —  Three-Pipeline AGO
======================================
Independent per-component AGO pipelines for VLM-decomposed descriptions.

Unlike ``ago_module`` which fuses all components into one ``ago_positive_prompt``
embedding, this module runs **three separate SDS optimisation pipelines**:

1. **Background pipeline** — optimises ``background_prompt`` to preserve
   normal product appearance.
2. **Defect pipeline** — optimises ``defect_prompt`` to capture anomaly
   morphology, texture, and severity.
3. **Position pipeline** — optimises ``position_prompt`` to encode spatial
   localisation of the defect relative to the object.

Each pipeline produces an independent ``[1, 77, 768]`` embedding. During
generation, the three embeddings can be used for spatially-aware
cross-attention control: background embedding for normal regions, defect
embedding for the anomalous region, and position embedding for spatial guidance.

Core classes:
- SinglePipelineAGO : SDS optimiser for one semantic component
- TripleAGO        : orchestrates all three pipelines
"""

from .triple_ago import SinglePipelineAGO, TripleAGO

__all__ = ["SinglePipelineAGO", "TripleAGO"]
