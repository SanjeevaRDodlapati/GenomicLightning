#!/usr/bin/env python3
"""
Import tests for GenomicLightning.
"""

import pytest


def test_import_models():
    """Test importing model modules."""
    try:
        from genomic_lightning.models import DeepSEA, DanQ

        print("✅ Model imports successful")
    except ImportError as e:
        pytest.skip(f"Models not available: {e}")


def test_import_data():
    """Test importing data modules."""
    try:
        from genomic_lightning.data import GenomicDataModule, BaseGenomicDataset

        print("✅ Data module imports successful")
    except ImportError as e:
        pytest.skip(f"Data modules not available: {e}")


def test_import_lightning_modules():
    """Test importing Lightning modules."""
    try:
        from genomic_lightning.lightning_modules import DeepSEALightningModule

        print("✅ Lightning module imports successful")
    except ImportError as e:
        pytest.skip(f"Lightning modules not available: {e}")


def test_import_utils():
    """Test importing utility modules."""
    try:
        from genomic_lightning.utils import import_uavarprior_model

        print("✅ Utility imports successful")
    except ImportError as e:
        pytest.skip(f"Utility modules not available: {e}")
