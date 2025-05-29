"""Custom callbacks for genomic deep learning training."""

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning.callbacks.base import Callback
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GenomicModelCheckpoint(ModelCheckpoint):
    """Model checkpoint callback optimized for genomic models."""

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: str = "val_auroc",
        mode: str = "max",
        save_top_k: int = 3,
        save_weights_only: bool = False,
        every_n_epochs: int = 1,
        **kwargs,
    ):
        """Initialize genomic model checkpoint.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor
            mode: Min or max optimization
            save_top_k: Number of best models to save
            save_weights_only: Save only model weights
            every_n_epochs: Save frequency
            **kwargs: Additional ModelCheckpoint arguments
        """

        # Default filename pattern for genomic models
        if filename is None:
            filename = "genomic-{epoch:02d}-{val_auroc:.4f}"

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            every_n_epochs=every_n_epochs,
            **kwargs,
        )

    def on_validation_end(self, trainer, pl_module):
        """Custom behavior after validation."""
        super().on_validation_end(trainer, pl_module)

        # Log checkpoint info
        if self.best_model_path:
            logger.info(f"Best model saved: {self.best_model_path}")
            logger.info(f"Best {self.monitor}: {self.best_model_score:.4f}")


class GenomicEarlyStopping(EarlyStopping):
    """Early stopping callback for genomic models."""

    def __init__(
        self,
        monitor: str = "val_auroc",
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 0.001,
        **kwargs,
    ):
        """Initialize genomic early stopping.

        Args:
            monitor: Metric to monitor
            mode: Min or max optimization
            patience: Number of epochs to wait
            min_delta: Minimum change threshold
            **kwargs: Additional EarlyStopping arguments
        """
        super().__init__(
            monitor=monitor, mode=mode, patience=patience, min_delta=min_delta, **kwargs
        )

    def on_validation_end(self, trainer, pl_module):
        """Custom behavior on validation end."""
        super().on_validation_end(trainer, pl_module)

        # Log early stopping status
        if self.wait_count > 0:
            logger.info(f"Early stopping patience: {self.wait_count}/{self.patience}")


class GenomicProgressBar(ProgressBar):
    """Progress bar with genomic-specific metrics display."""

    def __init__(self, refresh_rate: int = 1):
        """Initialize genomic progress bar."""
        super().__init__(refresh_rate=refresh_rate)

    def get_metrics(self, trainer, pl_module):
        """Get metrics to display in progress bar."""
        items = super().get_metrics(trainer, pl_module)

        # Add genomic-specific metrics if available
        if hasattr(pl_module, "last_auroc"):
            items["auroc"] = f"{pl_module.last_auroc:.3f}"
        if hasattr(pl_module, "last_auprc"):
            items["auprc"] = f"{pl_module.last_auprc:.3f}"

        return items


class VariantEffectLogger(Callback):
    """Callback to log variant effect predictions during training."""

    def __init__(self, log_every_n_epochs: int = 5, max_variants_to_log: int = 10):
        """Initialize variant effect logger.

        Args:
            log_every_n_epochs: Frequency of logging
            max_variants_to_log: Maximum number of variants to log
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.max_variants_to_log = max_variants_to_log

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log variant effects at end of validation epoch."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:

            # Check if model has variant effect prediction capability
            if hasattr(pl_module, "predict_variant_effects"):
                try:
                    # Get a small batch from validation set
                    val_loader = trainer.val_dataloaders
                    if val_loader and len(val_loader) > 0:
                        batch = next(iter(val_loader))
                        sequences, labels = batch[:2]  # Handle additional metadata

                        # Limit batch size for logging
                        n_samples = min(self.max_variants_to_log, sequences.size(0))
                        sequences = sequences[:n_samples].to(pl_module.device)

                        # Predict effects
                        with torch.no_grad():
                            effects = pl_module.predict_variant_effects(sequences)

                        # Log statistics
                        mean_effect = torch.mean(torch.abs(effects)).item()
                        max_effect = torch.max(torch.abs(effects)).item()

                        pl_module.log("variant_effect/mean_absolute", mean_effect)
                        pl_module.log("variant_effect/max_absolute", max_effect)

                        logger.info(
                            f"Epoch {trainer.current_epoch}: "
                            f"Mean |effect|: {mean_effect:.4f}, "
                            f"Max |effect|: {max_effect:.4f}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to log variant effects: {e}")


class SequenceVisualizationCallback(Callback):
    """Callback to visualize sequence attention/importance during training."""

    def __init__(
        self,
        output_dir: str = "sequence_visualizations",
        log_every_n_epochs: int = 10,
        max_sequences: int = 5,
    ):
        """Initialize sequence visualization callback.

        Args:
            output_dir: Directory to save visualizations
            log_every_n_epochs: Frequency of visualization
            max_sequences: Maximum sequences to visualize
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.log_every_n_epochs = log_every_n_epochs
        self.max_sequences = max_sequences

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Create sequence visualizations."""
        if trainer.current_epoch % self.log_every_n_epochs == 0:

            # Check if model supports attention/importance extraction
            if hasattr(pl_module, "get_sequence_importance"):
                try:
                    self._create_sequence_visualizations(trainer, pl_module)
                except Exception as e:
                    logger.warning(f"Failed to create sequence visualizations: {e}")

    def _create_sequence_visualizations(self, trainer, pl_module):
        """Create and save sequence visualizations."""

        # Get validation batch
        val_loader = trainer.val_dataloaders
        if not val_loader or len(val_loader) == 0:
            return

        batch = next(iter(val_loader))
        sequences, labels = batch[:2]

        # Limit number of sequences
        n_seq = min(self.max_sequences, sequences.size(0))
        sequences = sequences[:n_seq].to(pl_module.device)
        labels = labels[:n_seq]

        # Get sequence importance scores
        with torch.no_grad():
            importance_scores = pl_module.get_sequence_importance(sequences)

        # Create visualizations
        for i in range(n_seq):
            fig, ax = plt.subplots(figsize=(15, 4))

            # Plot importance scores
            seq_len = importance_scores.size(-1)
            positions = np.arange(seq_len)

            # Average importance across channels if needed
            if importance_scores.dim() == 3:  # [batch, channels, length]
                importance = importance_scores[i].mean(dim=0).cpu().numpy()
            else:  # [batch, length]
                importance = importance_scores[i].cpu().numpy()

            ax.plot(positions, importance, alpha=0.8)
            ax.fill_between(positions, importance, alpha=0.3)

            ax.set_xlabel("Sequence Position")
            ax.set_ylabel("Importance Score")
            ax.set_title(f"Sequence {i+1} Importance - Epoch {trainer.current_epoch}")

            # Add sequence annotation if available
            if hasattr(pl_module, "decode_sequence"):
                try:
                    seq_str = pl_module.decode_sequence(sequences[i])
                    # Show first 100 nucleotides as annotation
                    if len(seq_str) > 100:
                        seq_str = seq_str[:100] + "..."
                    ax.text(
                        0.02,
                        0.95,
                        f"Sequence: {seq_str}",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                except:
                    pass

            plt.tight_layout()

            # Save figure
            output_file = (
                self.output_dir / f"sequence_{i+1}_epoch_{trainer.current_epoch}.png"
            )
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

        logger.info(f"Saved {n_seq} sequence visualizations to {self.output_dir}")


class MetricsHistoryCallback(Callback):
    """Callback to track and save metrics history."""

    def __init__(
        self,
        save_path: Optional[str] = None,
        metrics_to_track: Optional[List[str]] = None,
    ):
        """Initialize metrics history callback.

        Args:
            save_path: Path to save metrics history
            metrics_to_track: List of metrics to track
        """
        super().__init__()
        self.save_path = save_path
        self.metrics_to_track = metrics_to_track or [
            "train_loss",
            "val_loss",
            "val_auroc",
            "val_auprc",
            "val_accuracy",
        ]
        self.history = {metric: [] for metric in self.metrics_to_track}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Record metrics at end of validation epoch."""

        # Get current metrics
        logged_metrics = trainer.logged_metrics

        for metric in self.metrics_to_track:
            if metric in logged_metrics:
                value = logged_metrics[metric].item()
                self.history[metric].append(value)
            else:
                self.history[metric].append(None)

    def on_train_end(self, trainer, pl_module):
        """Save metrics history at end of training."""
        if self.save_path:
            import json

            # Convert to serializable format
            serializable_history = {}
            for metric, values in self.history.items():
                serializable_history[metric] = [
                    float(v) if v is not None else None for v in values
                ]

            with open(self.save_path, "w") as f:
                json.dump(serializable_history, f, indent=2)

            logger.info(f"Metrics history saved to {self.save_path}")


def create_genomic_callbacks(config: Dict[str, Any]) -> List[Callback]:
    """Create standard set of callbacks for genomic training.

    Args:
        config: Training configuration

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint
    checkpoint_config = config.get("checkpoint", {})
    callbacks.append(
        GenomicModelCheckpoint(
            monitor=checkpoint_config.get("monitor", "val_auroc"),
            mode=checkpoint_config.get("mode", "max"),
            save_top_k=checkpoint_config.get("save_top_k", 3),
            dirpath=checkpoint_config.get("dirpath", "checkpoints/"),
        )
    )

    # Early stopping
    early_stopping_config = config.get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        callbacks.append(
            GenomicEarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_auroc"),
                patience=early_stopping_config.get("patience", 10),
                mode=early_stopping_config.get("mode", "max"),
            )
        )

    # Progress bar
    callbacks.append(GenomicProgressBar())

    # Variant effect logging
    if config.get("log_variant_effects", False):
        callbacks.append(
            VariantEffectLogger(
                log_every_n_epochs=config.get("variant_log_frequency", 5)
            )
        )

    # Sequence visualization
    viz_config = config.get("visualization", {})
    if viz_config.get("enabled", False):
        callbacks.append(
            SequenceVisualizationCallback(
                output_dir=viz_config.get("output_dir", "visualizations/"),
                log_every_n_epochs=viz_config.get("frequency", 10),
            )
        )

    # Metrics history
    if config.get("save_metrics_history", True):
        callbacks.append(
            MetricsHistoryCallback(
                save_path=config.get("metrics_history_path", "metrics_history.json")
            )
        )

    return callbacks
