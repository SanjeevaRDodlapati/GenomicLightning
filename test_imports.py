#!/usr/bin/env python3
"""Simple import test to identify potential issues."""

import sys
import traceback


def test_import(module_name):
    """Test importing a module and report any issues."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False


def main():
    print("🧪 Testing GenomicLightning imports...")
    print("=" * 50)

    modules_to_test = [
        "genomic_lightning",
        "genomic_lightning.models",
        "genomic_lightning.models.deepsea",
        "genomic_lightning.metrics",
        "genomic_lightning.metrics.genomic_metrics",
        "genomic_lightning.data",
        "genomic_lightning.lightning_modules",
    ]

    all_passed = True
    for module in modules_to_test:
        if not test_import(module):
            all_passed = False
        print()

    if all_passed:
        print("✅ All imports successful!")

        # Test basic functionality
        print("\n🔧 Testing basic functionality...")
        try:
            from genomic_lightning.models.deepsea import DeepSEA

            model = DeepSEA(sequence_length=1000, n_targets=919)
            print("✅ DeepSEA model creation successful")
        except Exception as e:
            print(f"❌ DeepSEA model creation failed: {e}")
            all_passed = False

        try:
            from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC

            metric = GenomicAUPRC(num_classes=3)
            print("✅ GenomicAUPRC creation successful")
        except Exception as e:
            print(f"❌ GenomicAUPRC creation failed: {e}")
            all_passed = False
    else:
        print("❌ Some imports failed!")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
