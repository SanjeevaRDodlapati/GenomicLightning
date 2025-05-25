"""Lightning data modules for genomic data."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

from .genomic_datasets import DeepSEADataset, GenomicSequenceDataset
from .base import BaseGenomicDataset

logger = logging.getLogger(__name__)


class GenomicDataModule(pl.LightningDataModule):
    """Lightning data module for genomic datasets."""
    
    def __init__(self,
                 train_data_path: Optional[str] = None,
                 val_data_path: Optional[str] = None,
                 test_data_path: Optional[str] = None,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 sequence_length: int = 1000,
                 num_classes: int = 919,
                 dataset_type: str = "deepsea",
                 train_val_split: float = 0.8,
                 **dataset_kwargs):
        """Initialize genomic data module.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data  
            test_data_path: Path to test data
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            sequence_length: Length of genomic sequences
            num_classes: Number of output classes
            dataset_type: Type of dataset ('deepsea', 'sequence')
            train_val_split: Train/validation split ratio if val_data_path not provided
            **dataset_kwargs: Additional arguments for dataset
        """
        super().__init__()
        
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.train_val_split = train_val_split
        self.dataset_kwargs = dataset_kwargs
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        
        if stage == "fit" or stage is None:
            # Setup training and validation datasets
            if self.train_data_path:
                if self.val_data_path:
                    # Separate train and val files
                    self.train_dataset = self._create_dataset(
                        self.train_data_path, split='train'
                    )
                    self.val_dataset = self._create_dataset(
                        self.val_data_path, split='val'
                    )
                else:
                    # Split single training file
                    full_dataset = self._create_dataset(
                        self.train_data_path, split='train'
                    )
                    
                    # Calculate split sizes
                    total_size = len(full_dataset)
                    train_size = int(self.train_val_split * total_size)
                    val_size = total_size - train_size
                    
                    self.train_dataset, self.val_dataset = random_split(
                        full_dataset, [train_size, val_size]
                    )
                    
                logger.info(f"Training samples: {len(self.train_dataset)}")
                logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            # Setup test dataset
            if self.test_data_path:
                self.test_dataset = self._create_dataset(
                    self.test_data_path, split='test'
                )
                logger.info(f"Test samples: {len(self.test_dataset)}")
        
        if stage == "predict":
            # For prediction, use test dataset or training dataset
            if self.test_data_path:
                self.test_dataset = self._create_dataset(
                    self.test_data_path, split='test'
                )
            elif self.train_data_path:
                self.test_dataset = self._create_dataset(
                    self.train_data_path, split='train'
                )
    
    def _create_dataset(self, data_path: str, split: str) -> BaseGenomicDataset:
        """Create dataset instance based on type."""
        
        if self.dataset_type == "deepsea":
            return DeepSEADataset(
                data_path=data_path,
                sequence_length=self.sequence_length,
                num_classes=self.num_classes,
                split=split,
                **self.dataset_kwargs
            )
        elif self.dataset_type == "sequence":
            # For sequence datasets, assume data_path is a FASTA file
            return GenomicSequenceDataset.from_fasta(
                fasta_path=data_path,
                sequence_length=self.sequence_length,
                **self.dataset_kwargs
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return prediction data loader."""
        return self.test_dataloader()
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "dataset_type": self.dataset_type
        }
        
        if self.train_dataset:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset:
            info["test_size"] = len(self.test_dataset)
        
        return info


class MultiModalGenomicDataModule(GenomicDataModule):
    """Data module for multi-modal genomic data (e.g., sequence + epigenomics)."""
    
    def __init__(self,
                 sequence_data_path: str,
                 epigenomic_data_path: Optional[str] = None,
                 **kwargs):
        """Initialize multi-modal data module.
        
        Args:
            sequence_data_path: Path to sequence data
            epigenomic_data_path: Path to epigenomic data
            **kwargs: Additional arguments for parent class
        """
        super().__init__(train_data_path=sequence_data_path, **kwargs)
        
        self.epigenomic_data_path = epigenomic_data_path
        
        # TODO: Implement multi-modal dataset loading
        # This would require custom datasets that can handle multiple data types
        if epigenomic_data_path:
            logger.warning("Multi-modal data loading not yet implemented")
    
    def setup(self, stage: Optional[str] = None):
        """Setup multi-modal datasets."""
        # For now, fall back to sequence-only datasets
        # TODO: Implement proper multi-modal dataset creation
        super().setup(stage)
        
        if self.epigenomic_data_path:
            logger.info(f"Epigenomic data path provided but not yet supported: "
                       f"{self.epigenomic_data_path}")


def create_data_module_from_config(config: Dict[str, Any]) -> GenomicDataModule:
    """Create data module from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GenomicDataModule instance
    """
    data_config = config.get("data", {})
    
    # Extract data module parameters
    dm_params = {
        "train_data_path": data_config.get("train_path"),
        "val_data_path": data_config.get("val_path"),
        "test_data_path": data_config.get("test_path"),
        "batch_size": data_config.get("batch_size", 64),
        "num_workers": data_config.get("num_workers", 4),
        "dataset_type": data_config.get("dataset_type", "deepsea")
    }
    
    # Add model-specific parameters
    model_config = config.get("model", {})
    dm_params.update({
        "sequence_length": model_config.get("input_length", 1000),
        "num_classes": model_config.get("num_classes", 919)
    })
    
    # Check for multi-modal data
    if data_config.get("epigenomic_path"):
        return MultiModalGenomicDataModule(
            sequence_data_path=dm_params["train_data_path"],
            epigenomic_data_path=data_config["epigenomic_path"],
            **{k: v for k, v in dm_params.items() if k != "train_data_path"}
        )
    else:
        return GenomicDataModule(**dm_params)