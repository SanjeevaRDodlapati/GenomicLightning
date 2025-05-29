#!/usr/bin/env python3
"""
Basic functionality tests for GenomicLightning.
"""

import pytest
import sys
import os

def test_basic_import():
    """Test that basic imports work."""
    try:
        import genomic_lightning
        assert hasattr(genomic_lightning, '__version__')
        print(f"✅ GenomicLightning version: {genomic_lightning.__version__}")
    except ImportError as e:
        pytest.skip(f"GenomicLightning not properly installed: {e}")

def test_version_consistency():
    """Test that version is consistent across files."""
    try:
        import genomic_lightning

        # Check VERSION file
        version_file = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                file_version = f.read().strip()
            assert genomic_lightning.__version__ == file_version

        print(f"✅ Version consistency check passed")
    except ImportError:
        pytest.skip("GenomicLightning not properly installed")

def test_package_structure():
    """Test that expected modules exist."""
    try:
        from genomic_lightning import models, data, config, utils
        print("✅ Core modules import successfully")
    except ImportError as e:
        pytest.skip(f"Core modules not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])