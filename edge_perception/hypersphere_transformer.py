"""
SSEE_3 Edge Perception: 256D Hypersphere & HMX Matrix Transformer.
Projects 256D embeddings onto 1-unit hypersphere surface (S^255) and reshapes to 16x16 HMX matrix tensors.
"""

from dataclasses import dataclass, asdict
import numpy as np
from typing import Dict, Any, Tuple


@dataclass
class HypersphereEmbeddingMatrix:
    """NPU-Ready Hypersphere Matrix Tensor."""
    embedding_dim: int  # 256
    matrix_shape: Tuple[int, int]  # (16, 16)
    npu_tensor_shape: Tuple[int, int, int, int]  # (1, 16, 16, 1)
    is_unit_hypersphere: bool
    l2_norm: float  # 1.0
    float32_matrix: np.ndarray
    int8_quantized_matrix: np.ndarray
    scale: float
    zero_point: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "matrix_shape": self.matrix_shape,
            "npu_tensor_shape": self.npu_tensor_shape,
            "is_unit_hypersphere": self.is_unit_hypersphere,
            "l2_norm": round(float(self.l2_norm), 6),
            "quantization_scale": self.scale,
            "quantization_zero_point": self.zero_point
        }


class HypersphereMatrixTransformer:
    """Projects 256D vector embeddings onto 1-unit hypersphere surface and builds HMX NPU matrix layout."""

    EMBEDDING_DIM = 256
    MATRIX_DIM = 16  # 16x16 = 256

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def project_and_transform(self, raw_embedding: np.ndarray) -> HypersphereEmbeddingMatrix:
        """Projects raw 256D embedding onto S^255 hypersphere surface and converts to 16x16 matrix."""
        vector = np.asarray(raw_embedding, dtype=np.float32).flatten()
        if vector.shape[0] != self.EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dimension {self.EMBEDDING_DIM}, got {vector.shape[0]}")

        # L2 Normalization -> S^255 Hypersphere Projection
        l2_norm = float(np.linalg.norm(vector))
        if l2_norm < self.eps:
            unit_vector = np.ones(self.EMBEDDING_DIM, dtype=np.float32) / np.sqrt(self.EMBEDDING_DIM)
        else:
            unit_vector = vector / (l2_norm + self.eps)

        final_norm = float(np.linalg.norm(unit_vector))
        is_unit = bool(np.isclose(final_norm, 1.0, atol=1e-4))

        matrix_2d = unit_vector.reshape((self.MATRIX_DIM, self.MATRIX_DIM))
        npu_tensor_4d = matrix_2d.reshape((1, self.MATRIX_DIM, self.MATRIX_DIM, 1))

        # Int8 Fixed-Point Quantization
        max_val = np.max(np.abs(unit_vector))
        scale = float(max_val / 127.0) if max_val > 0 else 1.0
        zero_point = 0
        int8_matrix = np.clip(np.round(matrix_2d / scale), -128, 127).astype(np.int8)

        return HypersphereEmbeddingMatrix(
            embedding_dim=self.EMBEDDING_DIM,
            matrix_shape=(self.MATRIX_DIM, self.MATRIX_DIM),
            npu_tensor_shape=(1, self.MATRIX_DIM, self.MATRIX_DIM, 1),
            is_unit_hypersphere=is_unit,
            l2_norm=final_norm,
            float32_matrix=matrix_2d,
            int8_quantized_matrix=int8_matrix,
            scale=scale,
            zero_point=zero_point
        )

    @staticmethod
    def compute_cosine_similarity(
        matrix_a: HypersphereEmbeddingMatrix,
        matrix_b: HypersphereEmbeddingMatrix
    ) -> float:
        """Computes NPU Dot-Product Similarity over 1-unit hypersphere matrices."""
        similarity = np.sum(matrix_a.float32_matrix * matrix_b.float32_matrix)
        return float(np.clip(similarity, -1.0, 1.0))
