"""
DeepSEA model implementation in PyTorch.

This module provides the DeepSEA model architecture for predicting chromatin features
from DNA sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DeepSEA(nn.Module):
    """
    DeepSEA model for predicting chromatin features from DNA sequences.

    Paper: Predicting effects of noncoding variants with deep learning-based sequence model
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        num_targets: int = 919,
        num_filters: List[int] = [320, 480, 960],
        filter_sizes: List[int] = [8, 8, 8],
        pool_sizes: List[int] = [4, 4, 4],
        dropout_rates: List[float] = [0.2, 0.2, 0.5],
        num_classes_per_target: int = 1,
    ):
        """
        Initialize the DeepSEA model.

        Args:
            sequence_length: Length of input DNA sequences
            num_targets: Number of chromatin features to predict
            num_filters: Number of filters for each convolutional layer
            filter_sizes: Filter sizes for each convolutional layer
            pool_sizes: Pool sizes for each pooling layer
            dropout_rates: Dropout rates for each layer
            num_classes_per_target: Number of classes per target (1 for binary classification)
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.num_targets = num_targets
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.pool_sizes = pool_sizes
        self.dropout_rates = dropout_rates
        self.num_classes_per_target = num_classes_per_target

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels = 4  # A, C, G, T

        for i, (num_filter, filter_size, pool_size, dropout_rate) in enumerate(
            zip(num_filters, filter_sizes, pool_sizes, dropout_rates)
        ):
            # Convolutional layer
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filter,
                kernel_size=filter_size,
                padding=filter_size // 2,
                bias=True,
            )
            self.conv_layers.append(conv)

            # Max pooling layer
            pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            self.pool_layers.append(pool)

            # Dropout layer
            dropout = nn.Dropout(dropout_rate)
            self.dropout_layers.append(dropout)

            in_channels = num_filter

        # Calculate the size after convolutions and pooling
        self._calculate_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 925)
        self.fc1_dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(925, 919)
        self.fc2_dropout = nn.Dropout(0.5)

        # Output layer
        if num_classes_per_target == 1:
            # Binary classification for each target
            self.output = nn.Linear(919, num_targets)
        else:
            # Multi-class classification for each target
            self.output = nn.Linear(919, num_targets * num_classes_per_target)

    def _calculate_conv_output_size(self):
        """Calculate the output size after all convolutional and pooling layers."""
        size = self.sequence_length

        for filter_size, pool_size in zip(self.filter_sizes, self.pool_sizes):
            # After convolution (with padding)
            size = size  # padding=filter_size//2 keeps size the same
            # After pooling
            size = size // pool_size

        self.conv_output_size = size * self.num_filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)

        Returns:
            Output tensor of shape (batch_size, num_targets) for binary classification
            or (batch_size, num_targets * num_classes_per_target) for multi-class
        """
        # Convolutional layers
        for conv, pool, dropout in zip(
            self.conv_layers, self.pool_layers, self.dropout_layers
        ):
            x = conv(x)
            x = F.relu(x)
            x = pool(x)
            x = dropout(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2_dropout(x)

        # Output layer
        x = self.output(x)

        if self.num_classes_per_target == 1:
            # Binary classification - apply sigmoid
            x = torch.sigmoid(x)
        else:
            # Multi-class classification - reshape and apply softmax
            x = x.view(-1, self.num_targets, self.num_classes_per_target)
            x = F.softmax(x, dim=-1)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for input sequences.

        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)

        Returns:
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels for input sequences.

        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)
            threshold: Threshold for binary classification

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(x)
        if self.num_classes_per_target == 1:
            return (proba > threshold).float()
        else:
            return torch.argmax(proba, dim=-1)


class DeepSEAVariant(DeepSEA):
    """
    DeepSEA model variant with different architecture choices.
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        num_targets: int = 919,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        **kwargs,
    ):
        """
        Initialize the DeepSEA variant model.

        Args:
            sequence_length: Length of input DNA sequences
            num_targets: Number of chromatin features to predict
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            sequence_length=sequence_length, num_targets=num_targets, **kwargs
        )

        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(num_filter) for num_filter in self.num_filters]
            )
        else:
            self.batch_norm_layers = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the variant model.

        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)

        Returns:
            Output tensor
        """
        # Store input for potential residual connections
        if self.use_residual:
            residual_inputs = []

        # Convolutional layers
        for i, (conv, pool, dropout) in enumerate(
            zip(self.conv_layers, self.pool_layers, self.dropout_layers)
        ):
            if self.use_residual:
                residual_inputs.append(x)

            x = conv(x)

            if self.use_batch_norm and self.batch_norm_layers is not None:
                x = self.batch_norm_layers[i](x)

            x = F.relu(x)

            # Add residual connection if dimensions match
            if self.use_residual and i > 0 and residual_inputs[-2].shape == x.shape:
                x = x + residual_inputs[-2]

            x = pool(x)
            x = dropout(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers (same as parent)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2_dropout(x)

        # Output layer
        x = self.output(x)

        if self.num_classes_per_target == 1:
            x = torch.sigmoid(x)
        else:
            x = x.view(-1, self.num_targets, self.num_classes_per_target)
            x = F.softmax(x, dim=-1)

        return x
