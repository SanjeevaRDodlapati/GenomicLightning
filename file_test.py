#!/usr/bin/env python3
"""Test script that writes results to file"""
import sys
import torch

def main():
    with open('/home/sdodl001/GenomicLightning/test_results.txt', 'w') as f:
        f.write("Testing GenomicLightning Metrics\n")
        f.write("="*40 + "\n")

        try:
            # Test imports
            from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC, TopKAccuracy, PositionalAUROC
            f.write("✅ Imports successful\n")

            # Test creation
            auprc = GenomicAUPRC(num_classes=3)
            f.write("✅ GenomicAUPRC created\n")

            topk = TopKAccuracy(k=5)
            f.write("✅ TopKAccuracy created\n")

            pos_auroc = PositionalAUROC(sequence_length=100, num_bins=5)
            f.write("✅ PositionalAUROC created\n")

            # Test basic functionality
            preds = torch.rand(5, 3)
            targets = torch.randint(0, 2, (5, 3)).float()

            auprc.update(preds, targets)
            result1 = auprc.compute()
            f.write(f"✅ GenomicAUPRC compute: {result1}\n")

            topk.update(preds, targets)
            result2 = topk.compute()
            f.write(f"✅ TopKAccuracy compute: {result2}\n")

            positions = torch.randint(0, 100, (5,))
            pos_auroc.update(preds, targets, positions)
            result3 = pos_auroc.compute()
            f.write(f"✅ PositionalAUROC compute keys: {list(result3.keys())}\n")

            f.write("\n🎉 ALL TESTS PASSED!\n")

        except Exception as e:
            f.write(f"❌ ERROR: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
