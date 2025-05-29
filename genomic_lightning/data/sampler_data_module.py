"""Data module using legacy samplers for backward compatibility."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union
import logging

from .sampler_adapter import SamplerAdapter
from ..utils.sampler_utils import SamplerUtils

logger = logging.getLogger(__name__)


class SamplerDataModule(pl.LightningDataModule):
    """Lightning data module that wraps legacy samplers."""

    def __init__(
        self,
        sampler_config: Dict[str, Any],
        batch_size: int = 64,
        num_workers: int = 4,
        use_legacy_sampler: bool = True,
        **kwargs,
    ):
        """Initialize sampler data module.

        Args:
            sampler_config: Configuration for the sampler
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            use_legacy_sampler: Whether to use legacy sampler wrapper
            **kwargs: Additional arguments
        """
        super().__init__()

        self.sampler_config = sampler_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_legacy_sampler = use_legacy_sampler

        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Legacy sampler wrapper
        self.sampler_utils = SamplerUtils()

        # Save hyperparameters
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        """Setup datasets using samplers."""

        if stage == "fit" or stage is None:
            # Setup training dataset
            train_sampler_config = self.sampler_config.get("train", {})
            if train_sampler_config:
                self.train_dataset = SamplerAdapter(
                    sampler_config=train_sampler_config,
                    sampler_utils=self.sampler_utils,
                    split="train",
                )
                logger.info(
                    f"Setup training dataset with {len(self.train_dataset)} samples"
                )

            # Setup validation dataset
            val_sampler_config = self.sampler_config.get("val", {})
            if val_sampler_config:
                self.val_dataset = SamplerAdapter(
                    sampler_config=val_sampler_config,
                    sampler_utils=self.sampler_utils,
                    split="val",
                )
                logger.info(
                    f"Setup validation dataset with {len(self.val_dataset)} samples"
                )

        if stage == "test" or stage is None:
            # Setup test dataset
            test_sampler_config = self.sampler_config.get("test", {})
            if test_sampler_config:
                self.test_dataset = SamplerAdapter(
                    sampler_config=test_sampler_config,
                    sampler_utils=self.sampler_utils,
                    split="test",
                )
                logger.info(f"Setup test dataset with {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            # Use custom collate function if available
            collate_fn=getattr(self.train_dataset, "collate_fn", None),
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not setup. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=getattr(self.val_dataset, "collate_fn", None),
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not setup. Call setup('test') first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=getattr(self.test_dataset, "collate_fn", None),
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction data loader."""
        return self.test_dataloader()

    def get_sample_data(self):
        """Get a sample batch for testing."""
        if self.train_dataset:
            return self.train_dataset[0]
        elif self.val_dataset:
            return self.val_dataset[0]
        elif self.test_dataset:
            return self.test_dataset[0]
        else:
            raise RuntimeError("No datasets available")


class UAVarPriorDataModule(SamplerDataModule):
    """Data module specifically for UAVarPrior samplers."""

    def __init__(self, uavarprior_config_path: str, **kwargs):
        """Initialize UAVarPrior data module.

        Args:
            uavarprior_config_path: Path to UAVarPrior configuration
            **kwargs: Additional arguments for parent class
        """

        # Load UAVarPrior configuration
        sampler_config = self._load_uavarprior_config(uavarprior_config_path)

        super().__init__(
            sampler_config=sampler_config, use_legacy_sampler=True, **kwargs
        )

        self.uavarprior_config_path = uavarprior_config_path

    def _load_uavarprior_config(self, config_path: str) -> Dict[str, Any]:
        """Load UAVarPrior configuration and convert to sampler config."""

        import yaml

        with open(config_path, "r") as f:
            uav_config = yaml.safe_load(f)

        # Convert UAVarPrior config to sampler config format
        sampler_config = {
            "train": {
                "type": "uavarprior",
                "config_path": config_path,
                "data_path": uav_config.get("train_data_path"),
                "split": "train",
            },
            "val": {
                "type": "uavarprior",
                "config_path": config_path,
                "data_path": uav_config.get("val_data_path"),
                "split": "val",
            },
            "test": {
                "type": "uavarprior",
                "config_path": config_path,
                "data_path": uav_config.get("test_data_path"),
                "split": "test",
            },
        }

        return sampler_config


class FuGEPDataModule(SamplerDataModule):
    """Data module specifically for FuGEP samplers."""

    def __init__(self, fugep_config_path: str, **kwargs):
        """Initialize FuGEP data module.

        Args:
            fugep_config_path: Path to FuGEP configuration
            **kwargs: Additional arguments for parent class
        """

        # Load FuGEP configuration
        sampler_config = self._load_fugep_config(fugep_config_path)

        super().__init__(
            sampler_config=sampler_config, use_legacy_sampler=True, **kwargs
        )

        self.fugep_config_path = fugep_config_path

    def _load_fugep_config(self, config_path: str) -> Dict[str, Any]:
        """Load FuGEP configuration and convert to sampler config."""

        import yaml

        with open(config_path, "r") as f:
            fugep_config = yaml.safe_load(f)

        # Convert FuGEP config to sampler config format
        sampler_config = {
            "train": {
                "type": "fugep",
                "config_path": config_path,
                "data_path": fugep_config.get("train_data"),
                "split": "train",
            },
            "val": {
                "type": "fugep",
                "config_path": config_path,
                "data_path": fugep_config.get("valid_data"),
                "split": "val",
            },
            "test": {
                "type": "fugep",
                "config_path": config_path,
                "data_path": fugep_config.get("test_data"),
                "split": "test",
            },
        }

        return sampler_config


def create_sampler_data_module(config: Dict[str, Any]) -> SamplerDataModule:
    """Create appropriate sampler data module from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SamplerDataModule instance
    """

    data_config = config.get("data", {})
    sampler_type = data_config.get("sampler_type", "generic")

    if sampler_type == "uavarprior":
        return UAVarPriorDataModule(
            uavarprior_config_path=data_config["config_path"],
            batch_size=data_config.get("batch_size", 64),
            num_workers=data_config.get("num_workers", 4),
        )

    elif sampler_type == "fugep":
        return FuGEPDataModule(
            fugep_config_path=data_config["config_path"],
            batch_size=data_config.get("batch_size", 64),
            num_workers=data_config.get("num_workers", 4),
        )

    else:
        # Generic sampler data module
        return SamplerDataModule(
            sampler_config=data_config.get("sampler_config", {}),
            batch_size=data_config.get("batch_size", 64),
            num_workers=data_config.get("num_workers", 4),
        )
