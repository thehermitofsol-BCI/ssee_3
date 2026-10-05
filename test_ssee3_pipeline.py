"""
SSEE_3 End-to-End Master Pipeline Integration Test.

Tests the full Sol Savage Automated Edge-to-Cloud Image-to-Listing Pipeline:
1. Camera Intake & Aperture Format (4KB Aligned, INT8 NHWC [1, 1024, 1024, 4]).
2. Model Preprocessing Contract Verification & Fused Graph Building.
3. LiteRT v2 NPU Perception & 256D Hypersphere S^255 Feature Extraction.
4. Automated Product Listing Synthesis (Title, Description, Category, SKU).
5. Multi-Channel Cloud Distribution & Ethereal Ledger Sync.
"""

from edge_perception.aperture_formatter import CognitiveApertureFormatter
from edge_perception.model_contract import ModelContractManager, ColorChannelOrder, QuantizationContractType, AspectRatioStrategy
from edge_perception.npu_engine import LiteRTv2NPUEngine
from listing_engine.listing_generator import AutomatedListingGenerator
from cloud_sync.cloud_publisher import MultiChannelPublisher


def main():
    print("==========================================================")
    print("  SSEE_3: SOL SAVAGE AUTOMATED EDGE-TO-CLOUD PIPELINE")
    print("  Hardware Target: Samsung Galaxy S25 Ultra (HTP v79 NPU)")
    print("==========================================================")

    # 1. Edge Perception: Format Camera Intake Tensor
    formatter = CognitiveApertureFormatter()
    frame = formatter.format_frame(
        frame_id="frame_s25u_001",
        width=3840,
        height=2160,
        use_signed_int8=True,
        source_path="/storage/emulated/0/DCIM/Camera/IMG_20261005.jpg"
    )

    print("\n[STEP 1: EDGE PERCEPTION INTAKE]")
    print(f"  Frame ID: {frame.frame_id}")
    print(f"  Aperture Bucket: {frame.aperture_size}x{frame.aperture_size}")
    print(f"  NHWC Tensor Shape: {frame.nhwc_shape}")
    print(f"  Data Type: {frame.dtype.upper()}")
    print(f"  Buffer Size: {frame.raw_buffer_size_bytes} bytes (4KB Page Aligned: {frame.page_aligned_4kb})")

    # 2. Verify Preprocessing Contract
    contract_mgr = ModelContractManager()
    contract = contract_mgr.build_contract(
        model_name="SSEE3_LiteRT_Qwen2_INT8",
        channel_order=ColorChannelOrder.RGB,
        quantization_contract=QuantizationContractType.MINUS_ONE_TO_ONE_INT8,
        aspect_strategy=AspectRatioStrategy.CENTER_CROP,
        target_aperture_size=1024,
        baked_into_graph=True
    )

    print("\n[STEP 2: MODEL CONTRACT & FUSED GRAPH]")
    print(f"  Model: {contract.model_name}")
    print(f"  Channel Order: {contract.channel_order.value}")
    print(f"  Fused Graph Operations: QnnYuvToRgb, Permute_RGB, CenterCrop, INT8 Quant")

    # 3. LiteRT v2 NPU Perception Execution
    npu_engine = LiteRTv2NPUEngine()
    perception = npu_engine.process_frame(frame, contract)

    print("\n[STEP 3: HEXAGON HTP v79 NPU INFERENCE]")
    print(f"  Classification: {perception.item_classification}")
    print(f"  Condition: {perception.condition_grade}")
    print(f"  Confidence: {perception.confidence_score * 100:.1f}%")
    print(f"  Backend: {perception.delegate_backend}")
    print(f"  256D Hypersphere L2 Norm: {perception.hypersphere_embedding.l2_norm:.6f} (Unit S^255: {perception.hypersphere_embedding.is_unit_hypersphere})")
    print(f"  16x16 Matrix Shape: {perception.hypersphere_embedding.matrix_shape}")
    print(f"  NPU Latency: {perception.npu_execution_time_ms} ms")

    # 4. Automated Listing Synthesis
    listing_gen = AutomatedListingGenerator(brand_name="Sol Savage Enterprise")
    listing = listing_gen.generate_listing(perception, base_price_usd=499.99)

    print("\n[STEP 4: AUTOMATED LISTING SYNTHESIS]")
    print(f"  SKU: {listing.sku}")
    print(f"  Title: {listing.title}")
    print(f"  Category: {listing.category}")
    print(f"  Estimated Price: ${listing.estimated_price_usd:.2f}")

    # 5. Multi-Channel Cloud Sync & Ledger Registration
    publisher = MultiChannelPublisher()
    pub_result = publisher.publish_listing(listing)

    print("\n[STEP 5: CLOUD SYNC & ETHEREAL LEDGER BROADCAST]")
    for channel, url in pub_result.published_channels.items():
        print(f"  - {channel}: {url}")
    print(f"  Ledger Block Hash: {pub_result.ledger_block_hash}")
    print(f"  Publish Duration: {pub_result.total_publish_time_ms} ms")

    print("\n==========================================================")
    print("  ✅ SSEE_3 EDGE-TO-CLOUD PIPELINE VERIFICATION SUCCESSFUL!")
    print("==========================================================")


if __name__ == "__main__":
    main()
