"""Adapter to wrap legacy samplers as PyTorch datasets."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging

from ..utils.sampler_utils import SamplerUtils

logger = logging.getLogger(__name__)


class SamplerAdapter(Dataset):
    """Adapter that wraps legacy samplers to work with PyTorch DataLoader."""
    
    def __init__(self,
                 sampler_config: Dict[str, Any],
                 sampler_utils: SamplerUtils,
                 split: str = "train",
                 transform: Optional[callable] = None):
        """Initialize sampler adapter.
        
        Args:
            sampler_config: Configuration for the legacy sampler
            sampler_utils: SamplerUtils instance for sampler management
            split: Data split ('train', 'val', 'test')
            transform: Optional transform to apply to data
        """
        super().__init__()
        
        self.sampler_config = sampler_config
        self.sampler_utils = sampler_utils
        self.split = split
        self.transform = transform
        
        # Initialize legacy sampler
        self.legacy_sampler = None
        self.dataset_size = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the legacy sampler."""
        
        try:
            sampler_type = self.sampler_config.get("type", "generic")
            
            if sampler_type == "uavarprior":
                self.legacy_sampler = self.sampler_utils.create_uavarprior_sampler(
                    config_path=self.sampler_config.get("config_path"),
                    data_path=self.sampler_config.get("data_path"),
                    split=self.split
                )
            
            elif sampler_type == "fugep":
                self.legacy_sampler = self.sampler_utils.create_fugep_sampler(
                    config_path=self.sampler_config.get("config_path"),
                    data_path=self.sampler_config.get("data_path"),
                    split=self.split
                )
            
            else:
                # Generic sampler
                self.legacy_sampler = self.sampler_utils.create_generic_sampler(
                    **self.sampler_config
                )
            
            # Get dataset size
            if hasattr(self.legacy_sampler, '__len__'):
                self.dataset_size = len(self.legacy_sampler)
            elif hasattr(self.legacy_sampler, 'size'):
                self.dataset_size = self.legacy_sampler.size
            else:
                # Try to estimate size by calling the sampler
                try:
                    self.dataset_size = self._estimate_size()
                except:
                    logger.warning("Could not determine dataset size, using default")
                    self.dataset_size = 10000  # Default fallback
            
            logger.info(f"Initialized {sampler_type} sampler for {self.split} "
                       f"with {self.dataset_size} samples")
        
        except Exception as e:
            logger.error(f"Failed to initialize sampler: {e}")
            raise
    
    def _estimate_size(self) -> int:
        """Estimate dataset size by probing the sampler."""
        
        # Try to call the sampler a few times to see if it has consistent behavior
        try:
            for i in range(10):
                sample = self.legacy_sampler[i]
                if sample is None:
                    return i
        except (IndexError, StopIteration):
            return i
        except Exception:
            pass
        
        # If we can't estimate, return a large number
        return 100000
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from legacy sampler."""
        
        try:
            # Call legacy sampler
            if hasattr(self.legacy_sampler, '__getitem__'):
                data = self.legacy_sampler[idx]
            elif hasattr(self.legacy_sampler, 'sample'):
                data = self.legacy_sampler.sample(idx)
            elif callable(self.legacy_sampler):
                data = self.legacy_sampler(idx)
            else:
                raise ValueError("Sampler is not callable or indexable")
            
            # Convert data to PyTorch tensors
            sequence, label = self._convert_data(data)
            
            # Apply transform if provided
            if self.transform:
                sequence = self.transform(sequence)
            
            return sequence, label
        
        except Exception as e:
            logger.warning(f"Error getting item {idx}: {e}")
            # Return dummy data to prevent training crashes
            return self._get_dummy_data()
    
    def _convert_data(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert legacy sampler output to PyTorch tensors."""
        
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            sequence, label = data[0], data[1]
        elif hasattr(data, 'sequence') and hasattr(data, 'label'):
            sequence, label = data.sequence, data.label
        elif isinstance(data, dict):
            sequence = data.get('sequence', data.get('X'))
            label = data.get('label', data.get('y'))
        else:
            # Assume single item is sequence, create dummy label
            sequence = data
            label = np.array([0.0])
        
        # Convert sequence to tensor
        if isinstance(sequence, str):
            # DNA sequence string - need to encode
            sequence = self._encode_dna_sequence(sequence)
        elif isinstance(sequence, np.ndarray):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        elif not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        
        # Convert label to tensor
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.float32)
        elif not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        
        return sequence, label
    
    def _encode_dna_sequence(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence string to one-hot tensor."""
        
        # DNA nucleotide mapping
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Convert to uppercase
        sequence = sequence.upper()
        
        # One-hot encoding
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in ['A', 'C', 'G', 'T']:
                idx = mapping[nucleotide]
                encoded[idx, i] = 1.0
            # Unknown nucleotides (N) remain as zeros
        
        return torch.tensor(encoded, dtype=torch.float32)
    
    def _get_dummy_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dummy data in case of errors."""
        
        # Create dummy sequence (1000bp, 4 channels for ACGT)
        dummy_sequence = torch.zeros(4, 1000, dtype=torch.float32)
        dummy_label = torch.zeros(1, dtype=torch.float32)
        
        return dummy_sequence, dummy_label
    
    def collate_fn(self, batch):
        """Custom collate function for handling variable-length sequences."""
        
        sequences, labels = zip(*batch)
        
        # Stack sequences
        try:
            sequences = torch.stack(sequences)
        except RuntimeError:
            # Handle variable length sequences by padding
            max_len = max(seq.size(-1) for seq in sequences)
            padded_sequences = []
            
            for seq in sequences:
                if seq.size(-1) < max_len:
                    # Pad sequence
                    pad_size = max_len - seq.size(-1)
                    if seq.dim() == 2:  # [channels, length]
                        padding = torch.zeros(seq.size(0), pad_size)
                        seq = torch.cat([seq, padding], dim=1)
                    else:  # [length]
                        padding = torch.zeros(pad_size)
                        seq = torch.cat([seq, padding], dim=0)
                
                padded_sequences.append(seq)
            
            sequences = torch.stack(padded_sequences)
        
        # Stack labels
        try:
            labels = torch.stack(labels)
        except RuntimeError:
            # Handle variable length labels
            max_len = max(label.numel() for label in labels)
            padded_labels = []
            
            for label in labels:
                if label.numel() < max_len:
                    pad_size = max_len - label.numel()
                    padding = torch.zeros(pad_size)
                    label = torch.cat([label.flatten(), padding])
                
                padded_labels.append(label)
            
            labels = torch.stack(padded_labels)
        
        return sequences, labels
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample if available."""
        
        try:
            if hasattr(self.legacy_sampler, 'get_metadata'):
                return self.legacy_sampler.get_metadata(idx)
            else:
                return {"sample_idx": idx, "split": self.split}
        except Exception as e:
            logger.warning(f"Could not get metadata for sample {idx}: {e}")
            return {"sample_idx": idx, "split": self.split}


class BatchedSamplerAdapter(SamplerAdapter):
    """Adapter for legacy samplers that return batches instead of single samples."""
    
    def __init__(self, 
                 sampler_config: Dict[str, Any],
                 sampler_utils: SamplerUtils,
                 batch_size: int = 64,
                 **kwargs):
        """Initialize batched sampler adapter.
        
        Args:
            sampler_config: Configuration for the legacy sampler
            sampler_utils: SamplerUtils instance
            batch_size: Batch size for legacy sampler
            **kwargs: Additional arguments for parent class
        """
        
        self.legacy_batch_size = batch_size
        self.current_batch = None
        self.current_batch_idx = 0
        
        super().__init__(sampler_config, sampler_utils, **kwargs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from batched legacy sampler."""
        
        # Calculate which batch and position within batch
        batch_idx = idx // self.legacy_batch_size
        within_batch_idx = idx % self.legacy_batch_size
        
        # Load new batch if needed
        if self.current_batch is None or batch_idx != self.current_batch_idx:
            self.current_batch = self._load_batch(batch_idx)
            self.current_batch_idx = batch_idx
        
        # Extract item from current batch
        if within_batch_idx < len(self.current_batch[0]):
            sequence = self.current_batch[0][within_batch_idx]
            label = self.current_batch[1][within_batch_idx]
            
            # Apply transform if provided
            if self.transform:
                sequence = self.transform(sequence)
            
            return sequence, label
        else:
            # Return dummy data if beyond batch size
            return self._get_dummy_data()
    
    def _load_batch(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch from the legacy sampler."""
        
        try:
            # Call legacy sampler to get batch
            if hasattr(self.legacy_sampler, 'get_batch'):
                batch_data = self.legacy_sampler.get_batch(batch_idx)
            elif hasattr(self.legacy_sampler, '__call__'):
                batch_data = self.legacy_sampler(batch_idx, self.legacy_batch_size)
            else:
                raise ValueError("Batched sampler does not support batch operations")
            
            # Convert batch data
            sequences, labels = self._convert_batch_data(batch_data)
            
            return sequences, labels
        
        except Exception as e:
            logger.warning(f"Error loading batch {batch_idx}: {e}")
            # Return dummy batch
            sequences = torch.zeros(self.legacy_batch_size, 4, 1000)
            labels = torch.zeros(self.legacy_batch_size, 1)
            return sequences, labels
    
    def _convert_batch_data(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert legacy batch data to PyTorch tensors."""
        
        if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
            sequences, labels = batch_data[0], batch_data[1]
        else:
            sequences = batch_data
            labels = np.zeros((len(sequences), 1))
        
        # Convert to tensors
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.tensor(sequences, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        
        return sequences, labels