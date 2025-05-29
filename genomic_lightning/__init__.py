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
from .utils import (
    import_uavarprior_model,
    import_fugep_model,
    import_model_from_path,
    GenericLightningWrapper,
    LegacySamplerWrapper,
    H5Dataset
)

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
    "import_uavarprior_model",
    "import_fugep_model",
    "import_model_from_path"
]