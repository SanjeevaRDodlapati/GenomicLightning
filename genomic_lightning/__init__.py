"""GenomicLightning: PyTorch Lightning framework for genomic deep learning."""

__version__ = "0.1.0"
__author__ = "Sanjeeva Dodlapati"
__email__ = "your.email@example.com"

# Core imports
from .models import DeepSEA, DanQ
from .lightning_modules import DeepSEALightningModule, DanQLightningModule
from .data import (
    BaseGenomicDataset,
    DeepSEADataset, 
    GenomicSequenceDataset,
    GenomicDataModule,
    ShardedGenomicDataModule
)
from .config import ConfigFactory, ConfigLoader
from .utils import LegacyModelImporter, LightningWrapper, SamplerUtils

__all__ = [
    "__version__",
    # Models
    "DeepSEA",
    "DanQ", 
    # Lightning modules
    "DeepSEALightningModule",
    "DanQLightningModule",
    # Data
    "BaseGenomicDataset",
    "DeepSEADataset",
    "GenomicSequenceDataset", 
    "GenomicDataModule",
    "ShardedGenomicDataModule",
    # Config
    "ConfigFactory",
    "ConfigLoader",
    # Utils
    "LegacyModelImporter",
    "LightningWrapper",
    "SamplerUtils"
]