"""
Base model class for genomic models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseGenomicModel(nn.Module, ABC):
    """
    Base class for all genomic models.

    This class provides common functionality and interface for genomic
    sequence prediction models.
    """

    def __init__(self, sequence_length: int = 1000, n_genomic_features: int = 4):
        """
        Initialize the base genomic model.

        Args:
            sequence_length: Length of input DNA sequences
            n_genomic_features: Number of features per position (4 for DNA: A,C,G,T)
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.n_genomic_features = n_genomic_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_genomic_features)

        Returns:
            Output predictions
        """
        pass

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture.

        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.__class__.__name__,
            "sequence_length": self.sequence_length,
            "n_genomic_features": self.n_genomic_features,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained weights from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            strict: Whether to strictly enforce matching keys
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present (from Lightning checkpoints)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                cleaned_state_dict[key[6:]] = value
            else:
                cleaned_state_dict[key] = value

        self.load_state_dict(cleaned_state_dict, strict=strict)

    def freeze_layers(self, layer_names: Optional[list] = None):
        """
        Freeze specified layers or all layers if none specified.

        Args:
            layer_names: List of layer names to freeze. If None, freeze all layers.
        """
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False

    def unfreeze_layers(self, layer_names: Optional[list] = None):
        """
        Unfreeze specified layers or all layers if none specified.

        Args:
            layer_names: List of layer names to unfreeze. If None, unfreeze all layers.
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True

    def get_layer_names(self) -> list:
        """
        Get names of all layers in the model.

        Returns:
            List of layer names
        """
        return [name for name, _ in self.named_modules()]

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
