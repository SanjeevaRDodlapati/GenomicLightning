#!/usr/bin/env python
"""Test script to check if genomic_lightning imports correctly."""

try:
    print("Testing genomic_lightning import...")
    import genomic_lightning
    print("✓ genomic_lightning imported successfully!")
    
    # Test basic components
    print("Testing model imports...")
    from genomic_lightning import DeepSEA, DanQ
    print("✓ Model classes imported successfully!")
    
    print("Testing lightning module imports...")
    from genomic_lightning import DeepSEALightningModule, DanQLightningModule
    print("✓ Lightning modules imported successfully!")
    
    print("Testing config imports...")
    from genomic_lightning import ConfigFactory, ConfigLoader
    print("✓ Config classes imported successfully!")
    
    print("Testing data imports...")
    from genomic_lightning import DeepSEADataset, GenomicDataModule
    print("✓ Data classes imported successfully!")
    
    print("All imports successful! 🎉")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
