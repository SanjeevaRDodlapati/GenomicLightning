#!/usr/bin/env python3

"""Debug script to test torchmetrics API."""

print("Starting torchmetrics compatibility test...")

try:
    import torch

    print(f"✅ PyTorch version: {torch.__version__}")

    import torchmetrics

    print(f"✅ TorchMetrics version: {torchmetrics.__version__}")

    # Test our custom metric
    print("\n🧪 Testing GenomicAUPRC creation...")
    from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC

    metric = GenomicAUPRC(num_classes=3)
    print("✅ GenomicAUPRC creation successful!")

    print("\n🧪 Testing TopKAccuracy creation...")
    from genomic_lightning.metrics.genomic_metrics import TopKAccuracy

    metric2 = TopKAccuracy(k=5)
    print("✅ TopKAccuracy creation successful!")

    print("\n🧪 Testing PositionalAUROC creation...")
    from genomic_lightning.metrics.genomic_metrics import PositionalAUROC

    metric3 = PositionalAUROC(sequence_length=1000, num_bins=10)
    print("✅ PositionalAUROC creation successful!")

    print("\n🎉 All metric creations successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
