"""Utility modules for GenomicLightning."""

from .legacy_import import import_uavarprior_model, import_fugep_model, import_model_from_path
from .wrapper_conversion import GenericLightningWrapper
from .sampler_utils import LegacySamplerWrapper

# Try to import H5Dataset - it might be from FuGEP's utils
try:
    from .prediction_utils import H5Dataset
except ImportError:
    # Fallback if H5Dataset is not defined in prediction_utils
    H5Dataset = None

__all__ = [
    "import_uavarprior_model", 
    "import_fugep_model", 
    "import_model_from_path",
    "GenericLightningWrapper",
    "LegacySamplerWrapper",
    "H5Dataset"
]