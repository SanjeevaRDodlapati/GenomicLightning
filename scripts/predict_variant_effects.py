#!/usr/bin/env python3
"""
Script to predict variant effects using trained genomic models.
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add GenomicLightning to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomic_lightning import ConfigFactory, ConfigLoader
from genomic_lightning.data import GenomicSequenceDataset
from genomic_lightning.utils import LegacyModelImporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict variant effects using genomic models"
    )
    
    parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--config-path",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--variants-file",
        required=True,
        help="CSV file with variant information (columns: ref_seq, alt_seq, variant_id)"
    )
    
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output CSV file for variant effect predictions"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (cpu, cuda, auto)"
    )
    
    parser.add_argument(
        "--model-type",
        default="lightning",
        choices=["lightning", "legacy"],
        help="Type of model to load"
    )
    
    return parser.parse_args()


def load_model(model_path, config_path=None, model_type="lightning", device="cpu"):
    """Load trained model."""
    
    if model_type == "lightning":
        # Load Lightning checkpoint
        if config_path:
            config = ConfigLoader.load_config(config_path)
            model = ConfigFactory.create_lightning_module(config)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try to load directly from checkpoint
            try:
                from genomic_lightning.lightning_modules import DeepSEALightning
                model = DeepSEALightning.load_from_checkpoint(
                    model_path, map_location=device
                )
            except Exception as e:
                logger.error(f"Failed to load Lightning model: {e}")
                raise
    
    elif model_type == "legacy":
        # Load legacy model (UAVarPrior/FuGEP)
        importer = LegacyModelImporter()
        model = importer.import_model(model_path, config_path)
        model = model.to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model


def load_variants(variants_file):
    """Load variant sequences from CSV file."""
    
    df = pd.read_csv(variants_file)
    
    required_columns = ['ref_seq', 'alt_seq']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Optional variant ID column
    if 'variant_id' not in df.columns:
        df['variant_id'] = [f"variant_{i}" for i in range(len(df))]
    
    return df


def predict_variant_effects(model, ref_sequences, alt_sequences, batch_size=32, device="cpu"):
    """Predict effects of variants."""
    
    # Create datasets for reference and alternative sequences
    ref_dataset = GenomicSequenceDataset(ref_sequences)
    alt_dataset = GenomicSequenceDataset(alt_sequences)
    
    ref_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=batch_size, shuffle=False
    )
    alt_loader = torch.utils.data.DataLoader(
        alt_dataset, batch_size=batch_size, shuffle=False
    )
    
    ref_predictions = []
    alt_predictions = []
    
    with torch.no_grad():
        # Predict for reference sequences
        for batch_seq, _ in ref_loader:
            batch_seq = batch_seq.to(device)
            pred = model(batch_seq)
            ref_predictions.append(pred.cpu())
        
        # Predict for alternative sequences
        for batch_seq, _ in alt_loader:
            batch_seq = batch_seq.to(device)
            pred = model(batch_seq)
            alt_predictions.append(pred.cpu())
    
    # Concatenate predictions
    ref_predictions = torch.cat(ref_predictions, dim=0)
    alt_predictions = torch.cat(alt_predictions, dim=0)
    
    # Calculate variant effects (difference between alt and ref)
    variant_effects = alt_predictions - ref_predictions
    
    return variant_effects.numpy()


def save_results(variant_effects, variant_ids, output_file):
    """Save variant effect predictions to CSV."""
    
    # Create DataFrame with results
    results_data = {
        'variant_id': variant_ids,
        'mean_effect': np.mean(np.abs(variant_effects), axis=1),
        'max_effect': np.max(np.abs(variant_effects), axis=1),
        'sum_positive_effects': np.sum(np.maximum(variant_effects, 0), axis=1),
        'sum_negative_effects': np.sum(np.minimum(variant_effects, 0), axis=1),
        'num_significant_effects': np.sum(np.abs(variant_effects) > 0.1, axis=1)
    }
    
    # Add individual feature effects if not too many features
    if variant_effects.shape[1] <= 100:
        for i in range(variant_effects.shape[1]):
            results_data[f'feature_{i}_effect'] = variant_effects[:, i]
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    logger.info(f"Variant effect predictions saved to {output_file}")
    
    # Print summary statistics
    logger.info(f"Processed {len(variant_ids)} variants")
    logger.info(f"Mean absolute effect: {np.mean(results_data['mean_effect']):.4f}")
    logger.info(f"Max absolute effect: {np.max(results_data['max_effect']):.4f}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(
            args.model_path, 
            args.config_path, 
            args.model_type,
            device
        )
        
        # Load variants
        logger.info(f"Loading variants from {args.variants_file}")
        variants_df = load_variants(args.variants_file)
        
        # Predict variant effects
        logger.info(f"Predicting effects for {len(variants_df)} variants")
        variant_effects = predict_variant_effects(
            model,
            variants_df['ref_seq'].tolist(),
            variants_df['alt_seq'].tolist(),
            batch_size=args.batch_size,
            device=device
        )
        
        # Save results
        save_results(
            variant_effects,
            variants_df['variant_id'].tolist(),
            args.output_file
        )
        
        logger.info("Variant effect prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during variant effect prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()