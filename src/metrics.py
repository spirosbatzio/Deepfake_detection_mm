from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores) -> float:
    """Equal Error Rate from fake-class scores (label 1 = fake = positive)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)
