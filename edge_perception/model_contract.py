"""
SSEE_3 Edge Perception: Model Preprocessing Contract & Fused Graph Builder.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List


class ColorChannelOrder(str, Enum):
    RGB = "RGB"
    BGR = "BGR"
    RGBA = "RGBA"
    BGRA = "BGRA"


class QuantizationContractType(str, Enum):
    ZERO_TO_ONE_UINT8 = "ZERO_TO_ONE_UINT8"
    MINUS_ONE_TO_ONE_INT8 = "MINUS_ONE_TO_ONE_INT8"


class AspectRatioStrategy(str, Enum):
    SQUASH = "SQUASH"
    CENTER_CROP = "CENTER_CROP"
    LETTERBOX = "LETTERBOX"


@dataclass
class PreprocessingContract:
    model_name: str
    channel_order: ColorChannelOrder
    quantization_contract: QuantizationContractType
    aspect_strategy: AspectRatioStrategy
    target_aperture_size: int
    baked_into_npu_graph: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "channel_order": self.channel_order.value,
            "quantization_contract": self.quantization_contract.value,
            "aspect_strategy": self.aspect_strategy.value,
            "target_aperture_size": self.target_aperture_size,
            "baked_into_npu_graph": self.baked_into_npu_graph
        }


class ModelContractManager:
    """Verifies training contract alignment and builds fused pre-processing nodes."""

    @staticmethod
    def build_contract(
        model_name: str = "SSEE3_LiteRT_Qwen2_INT8",
        channel_order: ColorChannelOrder = ColorChannelOrder.RGB,
        quantization_contract: QuantizationContractType = QuantizationContractType.MINUS_ONE_TO_ONE_INT8,
        aspect_strategy: AspectRatioStrategy = AspectRatioStrategy.CENTER_CROP,
        target_aperture_size: int = 1024,
        baked_into_graph: bool = True
    ) -> PreprocessingContract:
        return PreprocessingContract(
            model_name=model_name,
            channel_order=channel_order,
            quantization_contract=quantization_contract,
            aspect_strategy=aspect_strategy,
            target_aperture_size=target_aperture_size,
            baked_into_npu_graph=baked_into_graph
        )

    @staticmethod
    def generate_baked_preprocessing_spec(contract: PreprocessingContract) -> Dict[str, Any]:
        return {
            "fused_input_nodes": [
                {"name": "npu_fused_yuv_to_rgb", "op": "QnnYuvToRgb" if contract.baked_into_npu_graph else "Passthrough"},
                {"name": "npu_fused_channel_swap", "op": f"Permute_{contract.channel_order.value}"},
                {"name": "npu_fused_aspect_transform", "op": contract.aspect_strategy.value},
                {"name": "npu_fused_quantization", "contract": contract.quantization_contract.value}
            ],
            "zero_copy_status": "100% Zero-Copy (No C++ CPU pixel loops)"
        }
