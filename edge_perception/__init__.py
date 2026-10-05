"""
SSEE_3 Edge Perception Subsystem.
Zero-Copy Hardware-Aligned NPU Perception for Snapdragon 8 Elite.
"""

from edge_perception.aperture_formatter import CognitiveApertureFormatter, ApertureFrame
from edge_perception.npu_engine import LiteRTv2NPUEngine, PerceptionResult
from edge_perception.hypersphere_transformer import HypersphereMatrixTransformer, HypersphereEmbeddingMatrix
from edge_perception.model_contract import ModelContractManager, PreprocessingContract

__all__ = [
    "CognitiveApertureFormatter",
    "ApertureFrame",
    "LiteRTv2NPUEngine",
    "PerceptionResult",
    "HypersphereMatrixTransformer",
    "HypersphereEmbeddingMatrix",
    "ModelContractManager",
    "PreprocessingContract",
]
