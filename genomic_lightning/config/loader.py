"""Configuration loader for GenomicLightning."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads configuration from various formats (JSON, YAML, Python dict)."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dict.
        
        Args:
            config_path: Path to config file or dict
            
        Returns:
            Configuration dictionary
        """
        if isinstance(config_path, dict):
            return config_path
            
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        suffix = config_path.suffix.lower()
        
        if suffix == '.json':
            return ConfigLoader._load_json(config_path)
        elif suffix in ['.yaml', '.yml']:
            return ConfigLoader._load_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    
    @staticmethod
    def _load_json(config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")
    
    @staticmethod
    def _load_yaml(config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Config saved to {output_path}")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for GenomicLightning."""
        return {
            "model": {
                "type": "deepsea",
                "input_length": 1000,
                "num_classes": 919,
                "conv_layers": [320, 320, 320],
                "conv_kernels": [8, 8, 8],
                "pool_kernels": [4, 4, 4],
                "fc_layers": [925],
                "dropout": 0.2
            },
            "data": {
                "batch_size": 64,
                "num_workers": 4,
                "train_path": None,
                "val_path": None,
                "test_path": None
            },
            "training": {
                "max_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "optimizer": "adam",
                "scheduler": "cosine",
                "gradient_clip_val": 1.0
            },
            "logging": {
                "log_every_n_steps": 50,
                "save_top_k": 3,
                "monitor": "val_auroc",
                "mode": "max"
            }
        }