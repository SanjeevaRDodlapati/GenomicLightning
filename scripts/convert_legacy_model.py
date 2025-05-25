#!/usr/bin/env python3
"""
Script to convert legacy genomic models (UAVarPrior/FuGEP) to Lightning format.
"""

import argparse
import sys
import torch
from pathlib import Path
import logging

# Add GenomicLightning to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomic_lightning import ConfigFactory, ConfigLoader
from genomic_lightning.utils import LegacyModelImporter, LightningWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert legacy genomic models to Lightning format"
    )
    
    parser.add_argument(
        "--legacy-model-path",
        required=True,
        help="Path to legacy model file"
    )
    
    parser.add_argument(
        "--legacy-config-path",
        help="Path to legacy model configuration"
    )
    
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for converted Lightning model"
    )
    
    parser.add_argument(
        "--output-config-path",
        help="Output path for Lightning configuration"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["uavarprior", "fugep", "deepsea", "danq"],
        help="Type of legacy model"
    )
    
    parser.add_argument(
        "--lightning-config",
        help="Path to Lightning configuration template"
    )
    
    parser.add_argument(
        "--test-conversion",
        action="store_true",
        help="Test the converted model with sample data"
    )
    
    return parser.parse_args()


def detect_model_type(model_path, config_path=None):
    """Automatically detect the type of legacy model."""
    
    model_path = Path(model_path)
    
    # Check file extension and path patterns
    if "uavarprior" in str(model_path).lower():
        return "uavarprior"
    elif "fugep" in str(model_path).lower():
        return "fugep"
    elif "deepsea" in str(model_path).lower():
        return "deepsea"
    elif "danq" in str(model_path).lower():
        return "danq"
    
    # Check config file if available
    if config_path:
        config_path = Path(config_path)
        config_content = config_path.read_text().lower()
        
        if "uavarprior" in config_content:
            return "uavarprior"
        elif "fugep" in config_content:
            return "fugep"
        elif "deepsea" in config_content:
            return "deepsea"
        elif "danq" in config_content:
            return "danq"
    
    logger.warning("Could not automatically detect model type")
    return None


def create_lightning_config(legacy_model, model_type, output_path=None):
    """Create Lightning configuration from legacy model."""
    
    # Base configuration template
    config = {
        "model": {
            "type": "deepsea",  # Default, will be updated
            "input_length": 1000,
            "num_classes": 919
        },
        "data": {
            "batch_size": 64,
            "num_workers": 4
        },
        "training": {
            "max_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "optimizer": "adam"
        }
    }
    
    # Update based on model type and architecture
    if hasattr(legacy_model, 'config'):
        legacy_config = legacy_model.config
        
        # Extract relevant parameters
        if hasattr(legacy_config, 'input_length'):
            config["model"]["input_length"] = legacy_config.input_length
        if hasattr(legacy_config, 'num_classes'):
            config["model"]["num_classes"] = legacy_config.num_classes
        if hasattr(legacy_config, 'learning_rate'):
            config["training"]["learning_rate"] = legacy_config.learning_rate
    
    # Model-specific configurations
    if model_type == "deepsea":
        config["model"]["type"] = "deepsea"
        config["model"]["conv_layers"] = [320, 320, 320]
        config["model"]["conv_kernels"] = [8, 8, 8]
        config["model"]["pool_kernels"] = [4, 4, 4]
        config["model"]["fc_layers"] = [925]
    
    elif model_type == "danq":
        config["model"]["type"] = "danq"
        config["model"]["conv_layers"] = [320]
        config["model"]["conv_kernels"] = [26]
        config["model"]["pool_kernels"] = [13]
        config["model"]["lstm_hidden"] = 320
        config["model"]["fc_layers"] = [925]
    
    # Save configuration if output path provided
    if output_path:
        ConfigLoader.save_config(config, output_path)
        logger.info(f"Lightning configuration saved to {output_path}")
    
    return config


def convert_model(legacy_model_path, legacy_config_path, model_type, 
                 lightning_config_path=None):
    """Convert legacy model to Lightning format."""
    
    # Import legacy model
    logger.info(f"Importing legacy model: {legacy_model_path}")
    importer = LegacyModelImporter()
    
    legacy_model = importer.import_model(
        model_path=legacy_model_path,
        config_path=legacy_config_path,
        model_type=model_type
    )
    
    # Create Lightning configuration
    if lightning_config_path and Path(lightning_config_path).exists():
        config = ConfigLoader.load_config(lightning_config_path)
    else:
        config = create_lightning_config(legacy_model, model_type)
    
    # Create Lightning wrapper
    logger.info("Creating Lightning wrapper")
    wrapper = LightningWrapper(
        model=legacy_model,
        learning_rate=config["training"]["learning_rate"],
        optimizer=config["training"]["optimizer"]
    )
    
    return wrapper, config


def test_converted_model(lightning_model, config):
    """Test the converted model with sample data."""
    
    logger.info("Testing converted model with sample data")
    
    # Create sample input
    batch_size = 2
    input_length = config["model"]["input_length"]
    num_channels = 4  # DNA nucleotides
    
    sample_input = torch.randn(batch_size, num_channels, input_length)
    
    try:
        # Test forward pass
        lightning_model.eval()
        with torch.no_grad():
            output = lightning_model(sample_input)
        
        logger.info(f"Test successful! Output shape: {output.shape}")
        logger.info(f"Expected output shape: ({batch_size}, {config['model']['num_classes']})")
        
        # Check output shape
        expected_classes = config["model"]["num_classes"]
        if output.shape == (batch_size, expected_classes):
            logger.info("✓ Output shape matches expected")
            return True
        else:
            logger.warning(f"✗ Output shape mismatch: got {output.shape}, "
                          f"expected ({batch_size}, {expected_classes})")
            return False
    
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Detect model type if not provided
        model_type = args.model_type
        if not model_type:
            model_type = detect_model_type(args.legacy_model_path, args.legacy_config_path)
            if not model_type:
                logger.error("Could not detect model type. Please specify --model-type")
                sys.exit(1)
            logger.info(f"Detected model type: {model_type}")
        
        # Convert model
        lightning_model, config = convert_model(
            args.legacy_model_path,
            args.legacy_config_path,
            model_type,
            args.lightning_config
        )
        
        # Test conversion if requested
        if args.test_conversion:
            success = test_converted_model(lightning_model, config)
            if not success:
                logger.warning("Model test failed, but proceeding with save")
        
        # Save converted model
        logger.info(f"Saving converted model to {args.output_path}")
        torch.save(lightning_model.state_dict(), args.output_path)
        
        # Save configuration
        if args.output_config_path:
            ConfigLoader.save_config(config, args.output_config_path)
        
        logger.info("Model conversion completed successfully!")
        
        # Print summary
        logger.info("\nConversion Summary:")
        logger.info(f"  Input model: {args.legacy_model_path}")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Output model: {args.output_path}")
        if args.output_config_path:
            logger.info(f"  Output config: {args.output_config_path}")
        
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()