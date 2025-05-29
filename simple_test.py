#!/usr/bin/env python3
# Simple test of metrics
print("Testing metrics...")
try:
    from genomic_lightning.metrics.genomic_metrics import PositionalAUROC
    metric = PositionalAUROC(sequence_length=1000, num_bins=10)
    print("SUCCESS: PositionalAUROC created")
except Exception as e:
    print(f"ERROR: {e}")
