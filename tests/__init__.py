"""Test suite for GenomicLightning."""

# Test configuration
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)