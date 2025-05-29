"""Base dataset classes for genomic data."""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod


class BaseGenomicDataset(Dataset, ABC):
    """Abstract base class for genomic datasets."""

    def __init__(self,
                 sequence_length: int = 1000,
                 one_hot_encode: bool = True,
                 transform: Optional[callable] = None):
        """Initialize base genomic dataset.

        Args:
            sequence_length: Length of genomic sequences
            one_hot_encode: Whether to one-hot encode sequences
            transform: Optional transform to apply to data
        """
        self.sequence_length = sequence_length
        self.one_hot_encode = one_hot_encode
        self.transform = transform

        # DNA nucleotide mapping
        self.nucleotide_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.reverse_mapping = {v: k for k, v in self.nucleotide_mapping.items()}

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item."""
        pass

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to tensor.

        Args:
            sequence: DNA sequence string

        Returns:
            Encoded sequence tensor
        """
        # Convert to uppercase and handle unknown nucleotides
        sequence = sequence.upper()

        if self.one_hot_encode:
            # One-hot encoding: [A, C, G, T] channels
            encoded = np.zeros((4, len(sequence)), dtype=np.float32)
            for i, nucleotide in enumerate(sequence):
                if nucleotide in ['A', 'C', 'G', 'T']:
                    idx = self.nucleotide_mapping[nucleotide]
                    encoded[idx, i] = 1.0
                # Unknown nucleotides (N) remain as zeros in all channels
        else:
            # Integer encoding
            encoded = np.array([
                self.nucleotide_mapping.get(nuc, 4) for nuc in sequence
            ], dtype=np.int64)

        return torch.tensor(encoded)

    def decode_sequence(self, encoded_seq: torch.Tensor) -> str:
        """Convert encoded sequence back to DNA string.

        Args:
            encoded_seq: Encoded sequence tensor

        Returns:
            DNA sequence string
        """
        if self.one_hot_encode:
            # Convert one-hot to indices
            indices = torch.argmax(encoded_seq, dim=0)
            # Handle all-zero positions (unknown nucleotides)
            max_vals = torch.max(encoded_seq, dim=0)[0]
            indices[max_vals == 0] = 4  # N for unknown
        else:
            indices = encoded_seq

        return ''.join([self.reverse_mapping.get(int(idx), 'N') for idx in indices])

    def pad_or_truncate(self, sequence: str) -> str:
        """Pad or truncate sequence to target length.

        Args:
            sequence: Input DNA sequence

        Returns:
            Sequence of target length
        """
        if len(sequence) < self.sequence_length:
            # Pad with N's
            padding = 'N' * (self.sequence_length - len(sequence))
            return sequence + padding
        elif len(sequence) > self.sequence_length:
            # Truncate from center
            start = (len(sequence) - self.sequence_length) // 2
            return sequence[start:start + self.sequence_length]
        else:
            return sequence

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "sequence_length": self.sequence_length,
            "one_hot_encode": self.one_hot_encode,
            "num_classes": getattr(self, 'num_classes', None),
            "dataset_size": len(self)
        }


class MemoryGenomicDataset(BaseGenomicDataset):
    """In-memory genomic dataset for small datasets."""

    def __init__(self,
                 sequences: list,
                 labels: Optional[list] = None,
                 **kwargs):
        """Initialize memory dataset.

        Args:
            sequences: List of DNA sequences
            labels: Optional list of labels
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)

        self.sequences = sequences
        self.labels = labels

        if labels is not None and len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have same length")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.pad_or_truncate(self.sequences[idx])
        encoded_seq = self.encode_sequence(sequence)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Return dummy label for unsupervised tasks
            label = torch.tensor(0.0, dtype=torch.float32)

        if self.transform:
            encoded_seq = self.transform(encoded_seq)

        return encoded_seq, label