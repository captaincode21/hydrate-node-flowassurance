import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, TARGET_COL


def split_cases(kept_cases):
    """
    Split kept cases into train / val / test.
    With 3 cases: [0]=train, [1]=val, [2]=test.
    With 4+ cases: all-but-last-two=train, second-last=val, last=test.
    """
    ids = [int(df["case_id"].iloc[0]) for df in kept_cases]

    if len(ids) < 3:
        raise ValueError(
            f"Need at least 3 non-spiky cases, got {len(ids)}. "
            "Lower SPIKE_ABS_THRESHOLD or review data."
        )

    if len(ids) == 3:
        train_ids, val_ids, test_ids = [ids[0]], [ids[1]], [ids[2]]
    else:
        train_ids = ids[:-2]
        val_ids   = [ids[-2]]
        test_ids  = [ids[-1]]

    def select(id_list):
        return [df.copy() for df in kept_cases if int(df["case_id"].iloc[0]) in id_list]

    return select(train_ids), select(val_ids), select(test_ids), train_ids, val_ids, test_ids


def fit_target_transform(train_cases):
    """
    Fit log1p + standardisation on training targets only.
    Returns (y_mean, y_stddev).
    """
    train_log = pd.concat(
        [np.log1p(df[TARGET_COL].clip(lower=0)) for df in train_cases], axis=0
    )
    y_mean   = float(train_log.mean())
    y_stddev = float(train_log.std())
    if y_stddev < 1e-8:
        y_stddev = 1.0
    return y_mean, y_stddev


def apply_target_transform(cases, y_mean, y_stddev):
    """Add 'target_model' column (log1p-standardised target) to each case."""
    out = []
    for df in cases:
        df = df.copy()
        y_log = np.log1p(df[TARGET_COL].clip(lower=0))
        df["target_model"] = (y_log - y_mean) / y_stddev
        out.append(df)
    return out


def fit_feature_scaler(train_cases):
    """Fit StandardScaler on training features."""
    scaler = StandardScaler()
    train_x = pd.concat([df[FEATURE_COLS] for df in train_cases], axis=0)
    scaler.fit(train_x)
    return scaler


def apply_feature_scaler(cases, scaler):
    """Apply a fitted StandardScaler to feature columns in-place."""
    out = []
    for df in cases:
        df = df.copy()
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
        out.append(df)
    return out
