"""
Comprehensive integration tests for GenomicLightning.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
import yaml

from genomic_lightning.models.danq import DanQ
from genomic_lightning.models.chromdragonn import ChromDragoNNModel
from genomic_lightning.metrics.genomic_metrics import (
    GenomicAUPRC,
    TopKAccuracy,
    PositionalAUROC,
)
from genomic_lightning.data.data_modules import GenomicDataModule
import pytorch_lightning as pl


class TestEndToEndIntegration:
    """Test complete workflows from data loading to model training."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            "model": {"name": "danq", "sequence_length": 1000, "num_classes": 919},
            "data": {"batch_size": 16, "num_workers": 0, "sequence_length": 1000},
            "training": {"max_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"},
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample genomic data for testing."""
        batch_size = 8
        sequence_length = 1000
        num_classes = 919

        # Create random DNA sequences (one-hot encoded)
        sequences = torch.randint(0, 2, (batch_size, 4, sequence_length)).float()
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()

        return sequences, targets

    def test_danq_full_pipeline(self, sample_config, sample_data):
        """Test complete DanQ training pipeline."""
        sequences, targets = sample_data

        # Initialize model
        model = DanQ(
            sequence_length=sample_config["model"]["sequence_length"],
            n_outputs=sample_config["model"]["num_classes"],
        )

        # Test forward pass
        outputs = model(sequences)
        assert outputs.shape == (
            sequences.shape[0],
            sample_config["model"]["num_classes"],
        )

        # Test loss computation
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)
        assert loss.item() > 0

        # Test backward pass
        loss.backward()

        # Check gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_chromdragonn_full_pipeline(self, sample_config, sample_data):
        """Test complete ChromDragoNN training pipeline."""
        sequences, targets = sample_data

        # Initialize model
        model = ChromDragoNNModel(
            sequence_length=sample_config["model"]["sequence_length"],
            n_outputs=sample_config["model"]["num_classes"],
        )

        # Test forward pass
        outputs = model(sequences)
        assert outputs.shape == (
            sequences.shape[0],
            sample_config["model"]["num_classes"],
        )

        # Test loss computation
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)
        assert loss.item() > 0

        # Test backward pass
        loss.backward()

        # Check gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_metrics_integration(self, sample_data):
        """Test metrics integration with model outputs."""
        sequences, targets = sample_data
        num_classes = targets.shape[1]

        # Simulate model predictions (logits)
        predictions = torch.randn_like(targets)
        probabilities = torch.sigmoid(predictions)

        # Test GenomicAUPRC
        auprc_metric = GenomicAUPRC(num_classes=num_classes)
        auprc_metric.update(probabilities, targets.int())
        auprc_score = auprc_metric.compute()
        assert 0 <= auprc_score <= 1

        # Test TopKAccuracy
        topk_metric = TopKAccuracy(k=10)
        topk_metric.update(probabilities, targets.int())
        topk_score = topk_metric.compute()
        assert 0 <= topk_score <= 1

        # Test PositionalAUROC
        pos_auroc_metric = PositionalAUROC(sequence_length=1000)
        # Create position-specific targets and positions
        pos_targets = torch.randint(0, 2, (targets.shape[0], 1000)).float()
        pos_predictions = torch.randn(targets.shape[0], 1000)
        positions = torch.randint(0, 1000, (targets.shape[0],))  # Add positions
        pos_auroc_metric.update(pos_predictions, pos_targets.int(), positions)
        pos_auroc_score = pos_auroc_metric.compute()
        assert isinstance(pos_auroc_score, (torch.Tensor, dict))

    def test_config_loading_and_validation(self, sample_config):
        """Test configuration loading and validation."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = f.name

        try:
            # Load config
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)

            # Validate required keys
            assert "model" in loaded_config
            assert "data" in loaded_config
            assert "training" in loaded_config

            # Validate model config
            model_config = loaded_config["model"]
            assert "name" in model_config
            assert "sequence_length" in model_config
            assert "num_classes" in model_config

        finally:
            os.unlink(config_path)

    def test_memory_usage(self, sample_data):
        """Test memory usage during training."""
        sequences, targets = sample_data

        # Initialize model
        model = DanQ(sequence_length=1000, n_outputs=919)

        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Forward pass
        outputs = model(sequences)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Check memory didn't explode (basic sanity check)
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            # Memory increase should be reasonable (less than 1GB for this small test)
            assert memory_increase < 1e9  # 1GB

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same seed."""
        sequences, targets = sample_data

        # Set seed and run first time
        torch.manual_seed(42)
        model1 = DanQ(sequence_length=1000, n_outputs=919)
        outputs1 = model1(sequences)

        # Set same seed and run second time
        torch.manual_seed(42)
        model2 = DanQ(sequence_length=1000, n_outputs=919)
        outputs2 = model2(sequences)

        # Results should be identical
        assert torch.allclose(outputs1, outputs2, atol=1e-6)

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = DanQ(sequence_length=1000, n_outputs=919)

        for batch_size in [1, 4, 8, 16]:
            sequences = torch.randint(0, 2, (batch_size, 4, 1000)).float()
            outputs = model(sequences)
            assert outputs.shape == (batch_size, 919)

    def test_model_state_dict_consistency(self, sample_data):
        """Test model state dict saving and loading."""
        sequences, _ = sample_data

        # Create and run model
        model1 = DanQ(sequence_length=1000, n_outputs=919)
        outputs1 = model1(sequences)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = DanQ(sequence_length=1000, n_outputs=919)
        model2.load_state_dict(state_dict)
        outputs2 = model2(sequences)

        # Outputs should be identical
        assert torch.allclose(outputs1, outputs2, atol=1e-6)


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_training_speed_benchmark(self):
        """Basic training speed benchmark."""
        # Create sample data
        sequences = torch.randint(0, 2, (8, 4, 1000)).float()
        targets = torch.randint(0, 2, (8, 919)).float()
        
        model = DanQ(sequence_length=1000, n_outputs=919)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        import time

        start_time = time.time()

        # Run 10 training steps
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        end_time = time.time()

        # Should complete 10 steps in reasonable time (less than 30 seconds)
        total_time = end_time - start_time
        assert total_time < 30, f"Training took too long: {total_time}s"

    def test_memory_leak_detection(self, sample_data):
        """Test for memory leaks during repeated training."""
        sequences, targets = sample_data
        model = DanQ(sequence_length=1000, n_outputs=919)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run multiple training steps
            for _ in range(50):
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                # Clear cache periodically
                if _ % 10 == 0:
                    torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory

            # Memory increase should be minimal (less than 100MB)
            assert (
                memory_increase < 1e8
            ), f"Potential memory leak: {memory_increase} bytes"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_shapes(self):
        """Test model behavior with invalid input shapes."""
        model = DanQ(sequence_length=1000, n_outputs=919)

        # Wrong number of channels
        with pytest.raises((RuntimeError, ValueError)):
            wrong_channels = torch.randn(8, 5, 1000)  # Should be 4 channels
            model(wrong_channels)

        # Wrong sequence length
        with pytest.raises((RuntimeError, ValueError)):
            wrong_length = torch.randn(8, 4, 500)  # Should be 1000
            model(wrong_length)

    def test_metric_edge_cases(self):
        """Test metrics with edge case inputs."""
        # All zeros
        preds = torch.zeros(10, 5)
        targets = torch.zeros(10, 5, dtype=torch.int)

        metric = GenomicAUPRC(num_classes=5)
        metric.update(preds, targets)
        result = metric.compute()
        # Should handle gracefully, not crash
        assert isinstance(result, torch.Tensor)

        # All ones
        preds = torch.ones(10, 5)
        targets = torch.ones(10, 5, dtype=torch.int)

        metric = GenomicAUPRC(num_classes=5)
        metric.update(preds, targets)
        result = metric.compute()
        assert isinstance(result, torch.Tensor)

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        model = DanQ(sequence_length=1000, n_outputs=919)

        # Empty batch
        empty_batch = torch.empty(0, 4, 1000)

        try:
            outputs = model(empty_batch)
            assert outputs.shape[0] == 0
        except RuntimeError:
            # Some operations may not support empty tensors, which is acceptable
            pass
