#!/usr/bin/env python3
"""Quick test to verify fixes."""


def test_basic_syntax():
    """Test that files can be imported without syntax errors."""
    print("🧪 Testing basic syntax and imports...")

    try:
        import genomic_lightning

        print("✅ genomic_lightning imported successfully")
    except Exception as e:
        print(f"❌ genomic_lightning import failed: {e}")
        return False

    try:
        from genomic_lightning.metrics.genomic_metrics import GenomicAUPRC

        metric = GenomicAUPRC(num_classes=3)
        print("✅ GenomicAUPRC created successfully")
    except Exception as e:
        print(f"❌ GenomicAUPRC creation failed: {e}")
        return False

    print("✅ Basic tests passed!")
    return True


if __name__ == "__main__":
    success = test_basic_syntax()
    if success:
        print("\n🚀 Ready to test flake8!")
    else:
        print("\n❌ Basic tests failed - more fixes needed")
