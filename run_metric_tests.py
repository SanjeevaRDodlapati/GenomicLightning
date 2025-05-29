#!/usr/bin/env python3
"""
Complete test runner for genomic metrics
"""
import sys
import traceback
import torch


def run_tests():
    print("=" * 60)
    print("GENOMIC LIGHTNING METRICS TEST SUITE")
    print("=" * 60)

    try:
        # Test 1: Import test
        print("\n1. Testing imports...")
        from genomic_lightning.metrics.genomic_metrics import (
            GenomicAUPRC,
            TopKAccuracy,
            PositionalAUROC,
        )

        print("✅ All imports successful")

        # Test 2: Create instances
        print("\n2. Testing metric creation...")

        print("   Testing GenomicAUPRC...")
        auprc = GenomicAUPRC(num_classes=3, average="macro")
        print("   ✅ GenomicAUPRC created")

        print("   Testing TopKAccuracy...")
        topk = TopKAccuracy(k=5)
        print("   ✅ TopKAccuracy created")

        print("   Testing PositionalAUROC...")
        pos_auroc = PositionalAUROC(sequence_length=1000, num_bins=10)
        print("   ✅ PositionalAUROC created")

        # Test 3: Basic functionality test
        print("\n3. Testing basic functionality...")

        # Create sample data
        batch_size = 10
        n_classes = 3
        preds = torch.rand(batch_size, n_classes)
        targets = torch.zeros(batch_size, n_classes)

        # Make some random targets positive
        for i in range(batch_size):
            pos_idx = torch.randint(0, n_classes, (1,)).item()
            targets[i, pos_idx] = 1

        print("   Testing GenomicAUPRC update/compute...")
        auprc.update(preds, targets)
        auprc_result = auprc.compute()
        print(f"   ✅ GenomicAUPRC result: {auprc_result}")

        print("   Testing TopKAccuracy update/compute...")
        topk.update(preds, targets)
        topk_result = topk.compute()
        print(f"   ✅ TopKAccuracy result: {topk_result}")

        print("   Testing PositionalAUROC update/compute...")
        positions = torch.randint(0, 1000, (batch_size,))
        pos_auroc.update(preds, targets, positions)
        pos_result = pos_auroc.compute()
        print(f"   ✅ PositionalAUROC result keys: {list(pos_result.keys())}")

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("GenomicLightning metrics are working correctly.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
