import math
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import device
from src.model import run_trajectory
from src.utils import tensor_to_numpy_safe, inverse_target


def predict_dataset(model, dataset, y_mean: float, y_stddev: float) -> pd.DataFrame:
    """
    Run inference over every trajectory in a dataset.

    Returns a DataFrame with columns:
        pred_model  : standardised prediction
        true_model  : standardised ground truth
        pred_target : physical-space prediction  [kg/m³]
        true_target : physical-space ground truth [kg/m³]
    (plus all original columns from the case DataFrame)
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for t_np, x_np, y_np, cid, df_raw in dataset:
            pred_model = tensor_to_numpy_safe(
                run_trajectory(model, t_np, x_np, y_np[0])
            )
            true_model = np.array(y_np.squeeze(-1), dtype=np.float32)

            pred_phys = inverse_target(pred_model, y_mean, y_stddev)
            true_phys = inverse_target(true_model, y_mean, y_stddev)

            out = df_raw.copy()
            out["pred_model"]  = pred_model
            out["true_model"]  = true_model
            out["pred_target"] = pred_phys
            out["true_target"] = true_phys
            rows.append(out)

    return pd.concat(rows, axis=0).reset_index(drop=True) if rows else pd.DataFrame()


def compute_metrics(df: pd.DataFrame, name: str = "SET") -> dict | None:
    """
    Print and return RMSE / MAE / R² for a results DataFrame.
    Returns None if the DataFrame is empty.
    """
    if df.empty:
        print(f"{name}: empty — no metrics computed")
        return None

    rmse = math.sqrt(mean_squared_error(df["true_target"], df["pred_target"]))
    mae  = mean_absolute_error(df["true_target"], df["pred_target"])
    r2   = r2_score(df["true_target"], df["pred_target"])

    print(f"\n{'='*30}")
    print(f"{name} METRICS")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.6f}")

    return {"set": name, "rmse": rmse, "mae": mae, "r2": r2}
