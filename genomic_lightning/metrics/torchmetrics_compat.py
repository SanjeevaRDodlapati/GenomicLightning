"""
Compatibility layer for different torchmetrics versions.
"""

import torch
from typing import Optional, Any

def create_auroc(**kwargs):
    """Create AUROC metric with version compatibility."""
    from torchmetrics import AUROC
    
    # Remove deprecated parameters
    clean_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['compute_on_step', 'dist_sync_on_step', 'process_group']}
    
    try:
        # Try new API (torchmetrics >= 0.11)
        return AUROC(task="binary", **clean_kwargs)
    except TypeError:
        try:
            # Try with num_classes for older versions
            return AUROC(**clean_kwargs)
        except TypeError:
            # Fallback to simplest form
            return AUROC()

def create_precision_recall_curve(num_classes=None, **kwargs):
    """Create PrecisionRecallCurve metric with version compatibility."""
    from torchmetrics import PrecisionRecallCurve
    
    # Remove deprecated parameters
    clean_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['compute_on_step', 'dist_sync_on_step', 'process_group']}
    
    try:
        # Try new API (torchmetrics >= 0.11)
        if num_classes and num_classes > 2:
            return PrecisionRecallCurve(task="multilabel", num_labels=num_classes, **clean_kwargs)
        else:
            return PrecisionRecallCurve(task="binary", **clean_kwargs)
    except TypeError:
        try:
            # Try with num_classes for older versions
            if num_classes:
                return PrecisionRecallCurve(num_classes=num_classes, **clean_kwargs)
            else:
                return PrecisionRecallCurve(**clean_kwargs)
        except TypeError:
            # Fallback to simplest form
            return PrecisionRecallCurve()
