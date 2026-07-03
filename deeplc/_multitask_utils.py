"""Helper functions for multitask model output handling."""

from __future__ import annotations

import numpy as np


def _best_correlating_head(predictions: np.ndarray, targets: np.ndarray) -> int:
    """Return the head index with highest valid Pearson correlation to targets."""
    best_idx = 0
    best_corr = float("-inf")

    for idx in range(predictions.shape[1]):
        pred_col = predictions[:, idx]
        mask = np.isfinite(pred_col) & np.isfinite(targets)
        if mask.sum() < 3:
            continue
        pred_masked = pred_col[mask]
        target_masked = targets[mask]
        if np.std(pred_masked) < 1e-8 or np.std(target_masked) < 1e-8:
            continue
        corr = np.corrcoef(pred_masked, target_masked)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_idx = idx

    return best_idx


def _is_multitask_output(predictions: np.ndarray) -> bool:
    return predictions.ndim == 2 and predictions.shape[1] > 1
