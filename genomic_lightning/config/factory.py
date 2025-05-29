"""Configuration factory for GenomicLightning models."""

from typing import Dict, Any, Optional
import torch.nn as nn
from pathlib import Path

from .loader import ConfigLoader
from ..models.deepsea import DeepSEA
from ..models.danq import DanQ
from ..lightning_modules.deepsea import DeepSEALightningModule
from ..lightning_modules.danq import DanQLightningModule


class ConfigFactory:
    """Factory for creating models and Lightning modules from configuration."""

    MODEL_REGISTRY = {"deepsea": DeepSEA, "danq": DanQ}

    LIGHTNING_REGISTRY = {
        "deepsea": DeepSEALightningModule,
        "danq": DanQLightningModule,
    }

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> nn.Module:
        """Create a PyTorch model from configuration.

        Args:
            config: Model configuration

        Returns:
            PyTorch model instance
        """
        model_config = config.get("model", {})
        model_type = model_config.get("type", "deepsea")

        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls.MODEL_REGISTRY.keys())}"
            )

        model_class = cls.MODEL_REGISTRY[model_type]

        # Remove type from config before passing to model
        model_params = {k: v for k, v in model_config.items() if k != "type"}

        return model_class(**model_params)

    @classmethod
    def create_lightning_module(
        cls, config: Dict[str, Any], model: Optional[nn.Module] = None
    ) -> nn.Module:
        """Create a Lightning module from configuration.

        Args:
            config: Full configuration including model and training params
            model: Optional pre-created model

        Returns:
            Lightning module instance
        """
        model_config = config.get("model", {})
        model_type = model_config.get("type", "deepsea")

        if model_type not in cls.LIGHTNING_REGISTRY:
            raise ValueError(
                f"Unknown Lightning module type: {model_type}. "
                f"Available: {list(cls.LIGHTNING_REGISTRY.keys())}"
            )

        lightning_class = cls.LIGHTNING_REGISTRY[model_type]

        # Create model if not provided
        if model is None:
            model = cls.create_model(config)

        # Get training config
        training_config = config.get("training", {})

        return lightning_class(
            model=model,
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 1e-4),
            optimizer=training_config.get("optimizer", "adam"),
            scheduler=training_config.get("scheduler", "cosine"),
        )

    @classmethod
    def from_config_file(cls, config_path: str, model_only: bool = False):
        """Create model or Lightning module from config file.

        Args:
            config_path: Path to configuration file
            model_only: If True, return only the model, not Lightning module

        Returns:
            Model or Lightning module
        """
        config = ConfigLoader.load_config(config_path)

        if model_only:
            return cls.create_model(config)
        else:
            return cls.create_lightning_module(config)

    @classmethod
    def get_model_types(cls) -> list:
        """Get available model types."""
        return list(cls.MODEL_REGISTRY.keys())

    @classmethod
    def register_model(
        cls, name: str, model_class: type, lightning_class: Optional[type] = None
    ):
        """Register a new model type.

        Args:
            name: Model type name
            model_class: PyTorch model class
            lightning_class: Optional Lightning module class
        """
        cls.MODEL_REGISTRY[name] = model_class
        if lightning_class:
            cls.LIGHTNING_REGISTRY[name] = lightning_class
