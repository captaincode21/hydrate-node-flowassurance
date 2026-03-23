import numpy as np
from config import FEATURE_COLS


class TrajectoryDataset:
    """
    Wraps a list of case DataFrames into (t, x, y, case_id, raw_df) tuples
    ready for the NODE training loop.

    Iterating yields:
        t_np   : float32 numpy array of timestamps [T]
        x_np   : float32 numpy array of features   [T, F]
        y_np   : float32 numpy array of target      [T, 1]
        cid    : int case ID
        df_raw : original DataFrame (kept for post-hoc analysis)
    """

    def __init__(self, cases, target_col_model="target_model"):
        self.items = []
        for df in cases:
            t   = df["time_s"].values.astype(np.float32)
            x   = df[FEATURE_COLS].values.astype(np.float32)
            y   = df[target_col_model].values.astype(np.float32).reshape(-1, 1)
            cid = int(df["case_id"].iloc[0])
            self.items.append((t, x, y, cid, df.copy()))

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for item in self.items:
            yield item
