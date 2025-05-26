# GenomicLightning

A PyTorch Lightning framework for genomic deep learning models, designed with modern software architecture principles.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5)](https://www.pytorchlightning.ai/)

## Features

- **Modern Architecture**: Modular, composable components built on PyTorch Lightning
- **Multiple Models**: Support for DeepSEA, DanQ, ChromDragoNN, and custom architectures
- **Efficient Data Handling**: Streaming and sharding for large genomic datasets
- **Interpretability**: Advanced visualization tools for model interpretation
- **Specialized Metrics**: Metrics designed for genomic applications
- **Variant Analysis**: Tools for predicting and analyzing variant effects
- **Legacy Integration**: Seamless import from UAVarPrior/FuGEP models

## 🔗 Multi-Account Repository Access

This repository is synchronized across multiple GitHub accounts for enhanced collaboration:

- **Primary**: [SanjeevaRDodlapati/GenomicLightning](https://github.com/SanjeevaRDodlapati/GenomicLightning)
- **Mirror 1**: [sdodlapati3/GenomicLightning](https://github.com/sdodlapati3/GenomicLightning)
- **Mirror 2**: [sdodlapa/GenomicLightning](https://github.com/sdodlapa/GenomicLightning)

All repositories are automatically synchronized. Clone from any account you have access to.

## Installation

```bash
# Clone from any synchronized repository
git clone git@github.com:SanjeevaRDodlapati/GenomicLightning.git
# OR: git clone git@github.com:sdodlapati3/GenomicLightning.git  
# OR: git clone git@github.com:sdodlapa/GenomicLightning.git
cd GenomicLightning

## 🚀 Quick Start (Multi-Account Workflow)

### For Maintainers with Multi-Account Access

```bash
# 1. Make changes to the code
# 2. Commit your changes
git add .
git commit -m "Add new genomic model architecture"

# 3. Push to all accounts instantly
./push_all.csh

# Your changes are now live on all three GitHub accounts!
```

### For Contributors

```bash
# Fork from any of the synchronized repositories
# Submit PRs to the primary repository (SanjeevaRDodlapati/GenomicLightning)
```

## Installation

```bash
# Development installation
pip install -e ".[dev]"

# With logging tools
pip install -e ".[logging]"

# With all optional dependencies
pip install -e ".[dev,logging]"
```

## Quick Start

```bash
# Train a model
genomic_lightning train configs/example_deepsea.yml

# Evaluate a model
genomic_lightning evaluate configs/example_deepsea.yml --ckpt path/to/checkpoint.ckpt

# Train on large sharded data
python examples/large_data_training_example.py --train_shards data/train/*.h5 --val_shards data/val/*.h5 --model_type danq

# Run with model interpretability
python examples/large_data_training_example.py --train_shards data/train/*.h5 --val_shards data/val/*.h5 --interpret
```

## Supported Models

GenomicLightning includes implementations of several state-of-the-art genomic deep learning models:

- **DeepSEA** - Convolutional neural network for predicting chromatin effects of sequence alterations
- **DanQ** - Hybrid CNN-RNN architecture that captures both local motifs and dependencies
- **ChromDragoNN** - Residual network architecture for predicting chromatin features from DNA sequence

## Large Data Support

For working with large genomic datasets, use the sharded data module:

```python
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataModule

data_module = ShardedGenomicDataModule(
    train_shards=["data/train/shard1.h5", "data/train/shard2.h5"],
    val_shards=["data/val/shard1.h5"],
    batch_size=32,
    cache_size=1000  # Number of samples to cache in memory
)
```

## Interpretability Tools

Visualize and interpret what your models have learned:

```python
from genomic_lightning.visualization.motif_visualization import MotifVisualizer

# Create visualizer
visualizer = MotifVisualizer(model)

# Extract and visualize motifs from convolutional filters
visualizer.save_filter_logos(model, output_dir="motifs")

# Generate attribution maps using integrated gradients
attributions = visualizer.get_integrated_gradients(sequences, target_class=0)
fig = visualizer.visualize_sequence_attribution(sequences[0], attributions[0])
```

## Project Structure

- `genomic_lightning/models/` - Neural network architectures
- `genomic_lightning/data/` - Data loading and processing
- `genomic_lightning/lightning_modules/` - PyTorch Lightning modules
- `genomic_lightning/callbacks/` - Custom Lightning callbacks
- `genomic_lightning/cli/` - Command line interface
- `genomic_lightning/config/` - Configuration utilities
- `genomic_lightning/utils/` - Shared utilities
- `genomic_lightning/metrics/` - Specialized metrics for genomic data
- `genomic_lightning/visualization/` - Tools for model interpretability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🛠️ Advanced Multi-Account Features

### Repository Synchronization System

GenomicLightning uses an advanced multi-account system that provides:

- **Real-time Synchronization**: Changes propagate across all accounts within seconds
- **Unified Branch Management**: All repositories use `main` as the primary branch
- **Cross-Account Administration**: Full admin access across all synchronized accounts
- **Automated Backup**: Triple redundancy across GitHub accounts

### Development Workflow for Genomic Research

```bash
# Typical genomic research workflow
git checkout -b experiment-new-model
# ... develop your new genomic model
git add .
git commit -m "Implement novel transformer-based genomic architecture"

# Test your model
python tests/test_functionality.py

# Push to all research accounts
./push_all.csh
```

### Integration with Genomic Ecosystem

This repository integrates seamlessly with:
- **UAVarPrior**: Uncertainty quantification for genomic predictions
- **FuGEP**: Functional genomics event prediction framework
- **Multi-account synchronization**: Ensures research continuity across platforms

### 🔬 Research Collaboration Guidelines

1. **Experiment Tracking**: Each model experiment should be properly documented
2. **Data Privacy**: Follow genomic data handling best practices
3. **Reproducibility**: Use configuration files for all experiments
4. **Cross-Platform Sync**: Changes automatically appear on all research accounts

### 🧬 Genomic-Specific Features

- **Large Dataset Handling**: Efficient processing of multi-gigabyte genomic files
- **Model Interpretability**: Advanced visualization for understanding genomic predictions
- **Variant Effect Prediction**: Specialized tools for analyzing genetic variants
- **Multi-Modal Integration**: Support for various genomic data types
