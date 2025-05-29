"""
Custom metrics for genomic model evaluation.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Union, Any
from torchmetrics import Metric
from .torchmetrics_compat import create_auroc, create_precision_recall_curve


class GenomicAUPRC(Metric):
    """
    Area Under the Precision-Recall Curve (AUPRC) for genomic predictions.
    
    This metric is particularly useful for imbalanced genomic classification tasks
    where the positive class is rare, which is common in genomics.
    """
    
    def __init__(
        self,
        num_classes: int,
        average: str = "macro",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize the AUPRC metric.
        
        Args:
            num_classes: Number of classes
            average: Method for averaging ('micro', 'macro', 'weighted', None)
            dist_sync_on_step: Sync distributed metrics on step
            process_group: DDP process group
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        
        self.num_classes = num_classes
        self.average = average
        
        # Initialize precision-recall curve - use compatibility layer
        self.pr_curve = create_precision_recall_curve(num_classes=num_classes)
        
        # Register state
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metric state with batch predictions and targets.
        
        Args:
            preds: Model predictions [batch_size, num_classes]
            targets: Ground truth [batch_size, num_classes]
        """
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
        # Convert targets to int for torchmetrics compatibility
        targets = targets.int()
        self.preds.append(preds)
        self.targets.append(targets)
        
    def compute(self) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute the AUPRC.
        
        Returns:
            AUPRC value(s)
        """
        # Concatenate all batches
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        # Use PR curve to get precision and recall values
        precision, recall, _ = self.pr_curve(preds, targets)
        
        # Calculate AUPRC for each class
        auprcs = []
        for i in range(self.num_classes):
            # For multilabel tasks, precision and recall are lists of tensors
            if isinstance(precision, list):
                prec_i = precision[i]
                rec_i = recall[i]
            else:
                # For binary tasks, they are 2D tensors
                prec_i = precision[i]
                rec_i = recall[i]
                
            # Calculate area using trapezoidal rule
            # Sort by recall (ascending) to ensure proper integration
            if len(rec_i) > 1:
                sorted_indices = torch.argsort(rec_i)
                sorted_recall = rec_i[sorted_indices]
                sorted_precision = prec_i[sorted_indices]
                auprc_value = torch.trapz(sorted_precision, sorted_recall).item()
                # Ensure the result is non-negative
                auprc_value = max(0.0, auprc_value)
            else:
                auprc_value = 0.0
            auprcs.append(auprc_value)
            
        # Average based on specified method
        if self.average == "micro":
            # Combine predictions across all classes
            all_preds = preds.view(-1)
            all_targets = targets.view(-1)
            pr_curve = create_precision_recall_curve(num_classes=1)
            precision, recall, _ = pr_curve(all_preds.unsqueeze(1), all_targets.unsqueeze(1))
            
            # Calculate micro-average AUPRC
            if isinstance(precision, list):
                prec = precision[0]
                rec = recall[0]
            else:
                prec = precision[0]
                rec = recall[0]
                
            if len(rec) > 1:
                sorted_indices = torch.argsort(rec)
                sorted_recall = rec[sorted_indices]
                sorted_precision = prec[sorted_indices]
                micro_auprc = torch.trapz(sorted_precision, sorted_recall)
                return torch.clamp(micro_auprc, min=0.0)
            else:
                return torch.tensor(0.0)
            
        elif self.average == "macro":
            # Simple average across classes
            return torch.tensor(np.mean(auprcs))
            
        elif self.average == "weighted":
            # Weight by class prevalence
            class_counts = torch.sum(targets, dim=0)
            total_samples = torch.sum(class_counts)
            weights = class_counts / total_samples
            return torch.sum(torch.tensor(auprcs) * weights)
            
        else:
            # Return per-class values
            return {i: auprc for i, auprc in enumerate(auprcs)}


class TopKAccuracy(Metric):
    """
    Top-K accuracy for genomic prediction evaluation.
    
    This metric determines whether the top K predictions include
    the actual binding sites.
    """
    
    def __init__(
        self,
        k: int = 5,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize the Top-K accuracy metric.
        
        Args:
            k: Number of top predictions to consider
            dist_sync_on_step: Sync distributed metrics on step
            process_group: DDP process group
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        
        self.k = k
        
        # Register states
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metric state with batch predictions and targets.
        
        Args:
            preds: Model predictions [batch_size, num_classes]
            targets: Ground truth [batch_size, num_classes]
        """
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
        
        # Find top-k indices for each sample
        _, top_k_indices = torch.topk(preds, self.k, dim=1)
        
        # Find positive classes in targets
        positive_indices = torch.nonzero(targets, as_tuple=True)
        
        # Group by sample index
        sample_to_positive = {}
        for sample_idx, class_idx in zip(positive_indices[0].tolist(), 
                                          positive_indices[1].tolist()):
            if sample_idx not in sample_to_positive:
                sample_to_positive[sample_idx] = []
            sample_to_positive[sample_idx].append(class_idx)
            
        # Check if top-k predictions contain any positive classes
        correct = 0
        for sample_idx in range(preds.shape[0]):
            if sample_idx in sample_to_positive:
                positives = set(sample_to_positive[sample_idx])
                sample_top_k = set(top_k_indices[sample_idx].tolist())
                
                # If any positive class is in top-k, count as correct
                if len(positives.intersection(sample_top_k)) > 0:
                    correct += 1
                    
        # Update states
        self.correct += torch.tensor(correct)
        self.total += torch.tensor(preds.shape[0])
        
    def compute(self) -> torch.Tensor:
        """
        Compute the Top-K accuracy.
        
        Returns:
            Top-K accuracy value
        """
        return self.correct.float() / self.total.float() if self.total > 0 else torch.tensor(0.0)


class PositionalAUROC(Metric):
    """
    Position-aware AUROC metric for genomic sequences.
    
    This metric evaluates prediction quality at different positions
    in the input sequence, which is useful for understanding
    positional biases in genomic models.
    """
    
    def __init__(
        self,
        sequence_length: int,
        num_bins: int = 10,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize the Positional AUROC metric.
        
        Args:
            sequence_length: Length of input sequence
            num_bins: Number of position bins to evaluate
            dist_sync_on_step: Sync distributed metrics on step
            process_group: DDP process group
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        
        self.sequence_length = sequence_length
        self.num_bins = num_bins
        self.bin_size = sequence_length // num_bins
        
        # Create AUROC metrics for each position bin - use compatibility layer
        self.auroc_metrics = torch.nn.ModuleList([
            create_auroc() for _ in range(num_bins)
        ])
        
        # Use regular Python lists instead of metric states for nested data
        # These will be managed manually since torchmetrics doesn't support nested states
        self.bin_preds = [[] for _ in range(num_bins)]
        self.bin_targets = [[] for _ in range(num_bins)]
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor, 
               positions: torch.Tensor) -> None:
        """
        Update the metric state with predictions, targets, and positions.
        
        Args:
            preds: Model predictions [batch_size, num_classes]
            targets: Ground truth [batch_size, num_classes]
            positions: Position information [batch_size]
        """
        for i in range(preds.shape[0]):
            # Determine position bin
            pos = positions[i].item()
            bin_idx = min(pos // self.bin_size, self.num_bins - 1)
            
            # Add to appropriate bin
            self.bin_preds[bin_idx].append(preds[i:i+1])
            self.bin_targets[bin_idx].append(targets[i:i+1])
        
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute AUROC for each position bin.
        
        Returns:
            Dictionary mapping bins to AUROC values
        """
        results = {}
        
        for bin_idx in range(self.num_bins):
            if not self.bin_preds[bin_idx]:
                # No data for this bin
                results[f"bin_{bin_idx}"] = torch.tensor(0.0)
                continue
                
            # Concatenate predictions and targets for this bin
            bin_preds = torch.cat(self.bin_preds[bin_idx], dim=0)
            bin_targets = torch.cat(self.bin_targets[bin_idx], dim=0)
            
            # Compute AUROC for this bin
            auroc = self.auroc_metrics[bin_idx](bin_preds, bin_targets)
            results[f"bin_{bin_idx}"] = auroc
            
        # Also compute overall average
        valid_bins = [v for v in results.values() if v > 0]
        results["average"] = torch.mean(torch.stack(valid_bins)) if valid_bins else torch.tensor(0.0)
        
        return results