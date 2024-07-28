import numpy as np
from ._isotonic import _inplace_contiguous_isotonic_regression

def isotonic_regression(y, sample_weight=None, y_min=None, y_max=None, increasing=True):
    order = np.s_[:] if increasing else np.s_[::-1]
    # y = as_float_array(y)  # avoid sklearn dependency; we always pass arrays
    y = np.array(y[order], dtype=y.dtype)
    if sample_weight is None: sample_weight = np.ones(len(y), dtype=y.dtype)
    else: sample_weight = np.array(sample_weight[order], dtype=y.dtype)
    _inplace_contiguous_isotonic_regression(y, sample_weight)
    if y_min is not None or y_max is not None:
        # Older versions of np.clip don't accept None as a bound, so use np.inf
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y[order]
