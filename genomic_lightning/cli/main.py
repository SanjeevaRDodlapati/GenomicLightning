"""
Main CLI for GenomicLightning.
"""

import click
import yaml
import os
import sys
from pathlib import Path

from genomic_lightning.config.loader import load_config
from genomic_lightning.utils.training import train_model
from genomic_lightning.utils.prediction import predict_with_model
from genomic_lightning.utils.legacy_import import import_model_from_path


@click.group()
@click.version_option()
def cli():
    """GenomicLightning: PyTorch Lightning framework for genomic deep learning."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--output-dir', '-o', default='./output',
              help='Output directory for results')
@click.option('--gpus', default=None, type=int,
              help='Number of GPUs to use')
@click.option('--max-epochs', default=None, type=int,
              help='Maximum number of epochs to train')
@click.option('--resume', default=None, type=click.Path(exists=True),
              help='Path to checkpoint to resume from')
def train(config, output_dir, gpus, max_epochs, resume):
    """Train a genomic deep learning model."""
    
    # Load configuration
    config_data = load_config(config)
    
    # Override config with command line arguments
    if gpus is not None:
        config_data['trainer']['devices'] = gpus
    if max_epochs is not None:
        config_data['trainer']['max_epochs'] = max_epochs
    if resume is not None:
        config_data['trainer']['resume_from_checkpoint'] = resume
    
    # Set output directory
    config_data['trainer']['default_root_dir'] = output_dir
    
    # Train the model
    trainer, model = train_model(config_data)
    
    click.echo(f"Training completed! Results saved in {output_dir}")


@cli.command()
@click.option('--model-path', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model checkpoint')
@click.option('--data-path', '-d', required=True, type=click.Path(exists=True),
              help='Path to input data file')
@click.option('--output-path', '-o', required=True,
              help='Path to save predictions')
@click.option('--batch-size', default=32, type=int,
              help='Batch size for prediction')
@click.option('--gpus', default=None, type=int,
              help='Number of GPUs to use')
def predict(model_path, data_path, output_path, batch_size, gpus):
    """Make predictions with a trained model."""
    
    predictions = predict_with_model(
        model_path=model_path,
        data_path=data_path,
        batch_size=batch_size,
        devices=gpus
    )
    
    # Save predictions
    import torch
    torch.save(predictions, output_path)
    
    click.echo(f"Predictions saved to {output_path}")


@cli.command()
@click.option('--legacy-model', '-l', required=True, type=click.Path(exists=True),
              help='Path to legacy model file')
@click.option('--model-type', '-t', required=True,
              choices=['deepsea', 'danq', 'chromdragonn', 'custom'],
              help='Type of legacy model')
@click.option('--config-path', '-c', type=click.Path(exists=True),
              help='Path to model configuration file')
@click.option('--output-path', '-o', required=True,
              help='Path to save converted model')
def convert(legacy_model, model_type, config_path, output_path):
    """Convert a legacy model to GenomicLightning format."""
    
    # Import the legacy model
    model = import_model_from_path(
        model_path=legacy_model,
        model_type=model_type,
        config_path=config_path
    )
    
    # Save as PyTorch model
    import torch
    torch.save(model.state_dict(), output_path)
    
    click.echo(f"Converted model saved to {output_path}")


@cli.command()
@click.option('--input-files', '-i', multiple=True, required=True,
              help='Input HDF5 files to shard (can specify multiple)')
@click.option('--output-dir', '-o', required=True,
              help='Output directory for shards')
@click.option('--shard-size', default=10000, type=int,
              help='Number of samples per shard')
@click.option('--shuffle/--no-shuffle', default=True,
              help='Whether to shuffle data before sharding')
@click.option('--compression', default='gzip',
              help='Compression to use for output files')
def shard(input_files, output_dir, shard_size, shuffle, compression):
    """Create sharded datasets from large HDF5 files."""
    
    from genomic_lightning.utils.sampler_utils import create_sharded_dataset
    
    input_files = list(input_files)
    
    shard_paths = create_sharded_dataset(
        input_files=input_files,
        output_dir=output_dir,
        shard_size=shard_size,
        shuffle=shuffle,
        compression=compression
    )
    
    click.echo(f"Created {len(shard_paths)} shards in {output_dir}")


@cli.command()
@click.option('--config-template', '-t', 
              type=click.Choice(['deepsea', 'danq', 'chromdragonn']),
              default='deepsea',
              help='Type of configuration template to create')
@click.option('--output-path', '-o', default='config.yml',
              help='Path to save configuration file')
def init_config(config_template, output_path):
    """Initialize a configuration file from a template."""
    
    templates = {
        'deepsea': {
            'model': {
                'name': 'deepsea',
                'sequence_length': 1000,
                'num_targets': 919,
                'num_filters': [320, 480, 960],
                'filter_sizes': [8, 8, 8],
                'pool_sizes': [4, 4, 4],
                'dropout_rates': [0.2, 0.2, 0.5]
            },
            'data': {
                'type': 'hdf5',
                'train_data': 'path/to/train.h5',
                'val_data': 'path/to/val.h5',
                'test_data': 'path/to/test.h5',
                'batch_size': 64,
                'num_workers': 4
            },
            'trainer': {
                'max_epochs': 100,
                'devices': 1,
                'accelerator': 'auto',
                'precision': '16-mixed'
            },
            'optimizer': {
                'name': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-6
            },
            'scheduler': {
                'name': 'cosine',
                'eta_min': 1e-6
            }
        },
        'danq': {
            'model': {
                'name': 'danq',
                'sequence_length': 1000,
                'num_targets': 919,
                'num_filters': 320,
                'filter_size': 26,
                'pool_size': 13,
                'lstm_hidden': 320,
                'lstm_layers': 1
            },
            'data': {
                'type': 'hdf5',
                'train_data': 'path/to/train.h5',
                'val_data': 'path/to/val.h5',
                'test_data': 'path/to/test.h5',
                'batch_size': 64,
                'num_workers': 4
            },
            'trainer': {
                'max_epochs': 100,
                'devices': 1,
                'accelerator': 'auto',
                'precision': '16-mixed'
            },
            'optimizer': {
                'name': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-6
            }
        },
        'chromdragonn': {
            'model': {
                'name': 'chromdragonn',
                'sequence_length': 1000,
                'num_targets': 919,
                'num_filters': 300,
                'filter_sizes': [10, 15, 20],
                'residual_blocks': 3
            },
            'data': {
                'type': 'hdf5',
                'train_data': 'path/to/train.h5',
                'val_data': 'path/to/val.h5',
                'test_data': 'path/to/test.h5',
                'batch_size': 64,
                'num_workers': 4
            },
            'trainer': {
                'max_epochs': 100,
                'devices': 1,
                'accelerator': 'auto',
                'precision': '16-mixed'
            },
            'optimizer': {
                'name': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-6
            }
        }
    }
    
    config = templates[config_template]
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration template saved to {output_path}")
    click.echo(f"Please edit the file to specify your data paths and other settings.")


if __name__ == '__main__':
    cli()