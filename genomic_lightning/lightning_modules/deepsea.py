"""
PyTorch Lightning module for DeepSEA model.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import torchmetrics

from genomic_lightning.models.deepsea import DeepSEAModel


class DeepSEALightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training DeepSEA models.
    """
    
    def __init__(
        self,
        sequence_length: int = 1000,
        num_targets: int = 919,
        num_filters: List[int] = [320, 480, 960],
        filter_sizes: List[int] = [8, 8, 8],
        pool_sizes: List[int] = [4, 4, 4],
        dropout_rates: List[float] = [0.2, 0.2, 0.5],
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        weight_decay: float = 1e-6,
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize the DeepSEA Lightning module.
        
        Args:
            sequence_length: Length of input DNA sequences
            num_targets: Number of chromatin features to predict
            num_filters: Number of filters for each convolutional layer
            filter_sizes: Filter sizes for each convolutional layer
            pool_sizes: Pool sizes for each pooling layer
            dropout_rates: Dropout rates for each layer
            learning_rate: Learning rate for optimization
            optimizer: Optimizer to use ('adam', 'sgd', 'adamw')
            scheduler: Learning rate scheduler ('cosine', 'step', 'plateau')
            weight_decay: Weight decay for regularization
            pos_weight: Positive class weights for handling class imbalance
            class_weights: Class weights for handling class imbalance
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize the model
        self.model = DeepSEAModel(
            sequence_length=sequence_length,
            num_targets=num_targets,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            pool_sizes=pool_sizes,
            dropout_rates=dropout_rates
        )
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.class_weights = class_weights
        
        # Initialize metrics
        self.train_auroc = torchmetrics.AUROC(task="binary", num_classes=num_targets, average="macro")
        self.val_auroc = torchmetrics.AUROC(task="binary", num_classes=num_targets, average="macro")
        self.test_auroc = torchmetrics.AUROC(task="binary", num_classes=num_targets, average="macro")
        
        self.train_auprc = torchmetrics.AveragePrecision(task="binary", num_classes=num_targets, average="macro")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary", num_classes=num_targets, average="macro")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary", num_classes=num_targets, average="macro")
        
        self.train_accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_targets, average="macro")
        self.val_accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_targets, average="macro")
        self.test_accuracy = torchmetrics.Accuracy(task="binary", num_classes=num_targets, average="macro")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate the loss function."""
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy(predictions, targets)
        
        return loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        sequences, targets = batch
        predictions = self.forward(sequences)
        loss = self._calculate_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        self.train_auroc(predictions, targets.int())
        self.train_auprc(predictions, targets.int())
        self.train_accuracy(predictions, targets.int())
        
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_auprc', self.train_auprc, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        sequences, targets = batch
        predictions = self.forward(sequences)
        loss = self._calculate_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        self.val_auroc(predictions, targets.int())
        self.val_auprc(predictions, targets.int())
        self.val_accuracy(predictions, targets.int())
        
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auprc', self.val_auprc, on_step=False, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        sequences, targets = batch
        predictions = self.forward(sequences)
        loss = self._calculate_loss(predictions, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Calculate and log metrics
        self.test_auroc(predictions, targets.int())
        self.test_auprc(predictions, targets.int())
        self.test_accuracy(predictions, targets.int())
        
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True)
        self.log('test_auprc', self.test_auprc, on_step=False, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        
        return {'test_loss': loss}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Create optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        if self.scheduler_name is None:
            return optimizer
        
        # Create scheduler
        if self.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        sequences, _ = batch
        return self.forward(sequences)