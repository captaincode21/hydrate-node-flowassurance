import numpy as np
import torch


def tensor_to_numpy_safe(x):
    """Safely convert a torch tensor to a float32 numpy array."""
    return np.array(x.detach().cpu().tolist(), dtype=np.float32)


def inverse_target(y_std, y_mean, y_stddev):
    """Reverse the log1p + standardisation applied to the target."""
    y_std = np.asarray(y_std)
    y_log = y_std * y_stddev + y_mean
    return np.expm1(y_log)


def trim_steady_tail(df, target, slope_thresh=1e-4, window=20, min_keep_frac=0.6):
    """
    Remove the flat trailing section of a time series.
    Cuts at the first point where the gradient stays below
    slope_thresh for `window` consecutive steps.
    """
    x = df[target].values
    t = df["time_s"].values

    if len(df) < window + 5:
        return df.copy()

    dxdt   = np.gradient(x, t)
    flat   = np.abs(dxdt) < slope_thresh
    run    = 0
    cut_idx = None
    min_idx = int(len(df) * min_keep_frac)

    for i in range(len(flat)):
        run = run + 1 if flat[i] else 0
        if i >= min_idx and run >= window:
            cut_idx = i
            break

    return df.iloc[:cut_idx + 1].copy() if cut_idx is not None else df.copy()


def detect_spiky_case(df, target, abs_thresh=80.0, ratio_thresh=3.0):
    """
    Flag a case as spiky if:
      - its max target exceeds abs_thresh, OR
      - (max / median of positive values) exceeds ratio_thresh
    Returns (is_spiky: bool, stats: dict).
    """
    y     = df[target].astype(float).values
    y_pos = y[y > 1e-8]

    if len(y_pos) == 0:
        return False, {"max": 0.0, "median_pos": 0.0, "ratio": 0.0}

    y_max  = float(np.max(y))
    y_med  = float(np.median(y_pos))
    ratio  = y_max / max(y_med, 1e-8)
    is_spiky = (y_max > abs_thresh) or (ratio > ratio_thresh)

    return is_spiky, {"max": y_max, "median_pos": y_med, "ratio": ratio}
