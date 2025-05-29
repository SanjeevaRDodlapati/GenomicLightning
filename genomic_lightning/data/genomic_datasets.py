"""Specific genomic dataset implementations."""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging

from .base import BaseGenomicDataset

logger = logging.getLogger(__name__)


class DeepSEADataset(BaseGenomicDataset):
    """Dataset for DeepSEA format data (sequences + chromatin features)."""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 1000,
        num_classes: int = 919,
        split: str = "train",
        **kwargs,
    ):
        """Initialize DeepSEA dataset.

        Args:
            data_path: Path to HDF5 data file
            sequence_length: Length of sequences
            num_classes: Number of chromatin features
            split: Data split ('train', 'val', 'test')
            **kwargs: Additional arguments for base class
        """
        super().__init__(sequence_length=sequence_length, **kwargs)

        self.data_path = Path(data_path)
        self.num_classes = num_classes
        self.split = split

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load data info
        self._load_data_info()

    def _load_data_info(self):
        """Load dataset information without loading full data."""
        with h5py.File(self.data_path, "r") as f:
            # Check available splits
            if self.split not in f.keys():
                available_splits = list(f.keys())
                raise ValueError(
                    f"Split '{self.split}' not found. " f"Available: {available_splits}"
                )

            split_group = f[self.split]

            # Get dataset size
            if "sequences" in split_group:
                self.dataset_size = split_group["sequences"].shape[0]
            elif "X" in split_group:
                self.dataset_size = split_group["X"].shape[0]
            else:
                raise KeyError(f"No sequence data found in split '{self.split}'")

            logger.info(f"Loaded {self.split} split with {self.dataset_size} samples")

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.data_path, "r") as f:
            split_group = f[self.split]

            # Load sequence
            if "sequences" in split_group:
                sequence_data = split_group["sequences"][idx]
            elif "X" in split_group:
                sequence_data = split_group["X"][idx]
            else:
                raise KeyError("No sequence data found")

            # Handle different sequence formats
            if isinstance(sequence_data, bytes):
                # String sequence
                sequence = sequence_data.decode("utf-8")
                encoded_seq = self.encode_sequence(sequence)
            elif isinstance(sequence_data, np.ndarray):
                if sequence_data.ndim == 1:
                    # Integer encoded sequence
                    encoded_seq = torch.tensor(sequence_data, dtype=torch.long)
                    if self.one_hot_encode:
                        # Convert to one-hot
                        one_hot = torch.zeros(4, len(sequence_data))
                        for i, nuc_idx in enumerate(sequence_data):
                            if 0 <= nuc_idx < 4:
                                one_hot[nuc_idx, i] = 1.0
                        encoded_seq = one_hot
                elif sequence_data.ndim == 2:
                    # Already one-hot encoded
                    encoded_seq = torch.tensor(sequence_data, dtype=torch.float32)
                else:
                    raise ValueError(
                        f"Unexpected sequence shape: {sequence_data.shape}"
                    )
            else:
                raise ValueError(f"Unexpected sequence type: {type(sequence_data)}")

            # Load labels
            if "labels" in split_group:
                labels = split_group["labels"][idx]
            elif "y" in split_group:
                labels = split_group["y"][idx]
            else:
                # Create dummy labels
                labels = np.zeros(self.num_classes, dtype=np.float32)

            labels = torch.tensor(labels, dtype=torch.float32)

            if self.transform:
                encoded_seq = self.transform(encoded_seq)

            return encoded_seq, labels


class GenomicSequenceDataset(BaseGenomicDataset):
    """Generic genomic sequence dataset for FASTA/text files."""

    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize genomic sequence dataset.

        Args:
            sequences: List of DNA sequences
            labels: Optional labels for each sequence
            metadata: Optional metadata dict
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)

        self.sequences = sequences
        self.labels = labels
        self.metadata = metadata or {}

        if labels is not None and len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have same length")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get and process sequence
        raw_sequence = self.sequences[idx]
        sequence = self.pad_or_truncate(raw_sequence)
        encoded_seq = self.encode_sequence(sequence)

        # Get labels
        if self.labels is not None:
            if isinstance(self.labels[idx], (list, np.ndarray)):
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
            else:
                label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        else:
            label = torch.tensor([0.0], dtype=torch.float32)

        if self.transform:
            encoded_seq = self.transform(encoded_seq)

        return encoded_seq, label

    @classmethod
    def from_fasta(cls, fasta_path: str, **kwargs):
        """Create dataset from FASTA file.

        Args:
            fasta_path: Path to FASTA file
            **kwargs: Additional arguments for dataset

        Returns:
            GenomicSequenceDataset instance
        """
        sequences = []
        sequence_ids = []

        current_seq = ""
        current_id = None

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence
                    if current_seq and current_id:
                        sequences.append(current_seq)
                        sequence_ids.append(current_id)

                    # Start new sequence
                    current_id = line[1:]  # Remove '>'
                    current_seq = ""
                else:
                    current_seq += line

            # Save last sequence
            if current_seq and current_id:
                sequences.append(current_seq)
                sequence_ids.append(current_id)

        metadata = {"sequence_ids": sequence_ids, "source": fasta_path}

        return cls(sequences=sequences, metadata=metadata, **kwargs)


class VariantEffectDataset(BaseGenomicDataset):
    """Dataset for variant effect prediction."""

    def __init__(
        self,
        reference_sequences: List[str],
        variant_sequences: List[str],
        variant_info: List[Dict],
        **kwargs,
    ):
        """Initialize variant effect dataset.

        Args:
            reference_sequences: Reference genome sequences
            variant_sequences: Sequences with variants
            variant_info: Information about each variant
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)

        self.reference_sequences = reference_sequences
        self.variant_sequences = variant_sequences
        self.variant_info = variant_info

        if not (
            len(reference_sequences) == len(variant_sequences) == len(variant_info)
        ):
            raise ValueError("All input lists must have same length")

    def __len__(self) -> int:
        return len(self.reference_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Encode reference and variant sequences
        ref_seq = self.pad_or_truncate(self.reference_sequences[idx])
        var_seq = self.pad_or_truncate(self.variant_sequences[idx])

        ref_encoded = self.encode_sequence(ref_seq)
        var_encoded = self.encode_sequence(var_seq)

        # Stack sequences (ref and variant as separate channels or batch)
        sequences = torch.stack([ref_encoded, var_encoded], dim=0)

        # Create dummy label (actual effects computed by model)
        label = torch.tensor([0.0], dtype=torch.float32)

        if self.transform:
            sequences = self.transform(sequences)

        return sequences, label, self.variant_info[idx]
