"""
main.py — entry point for the hydrate NODE training pipeline.

Run with:
    python main.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import OUTPUT_DIR, device
from src.data_loader    import load_all_cases
from src.preprocessing  import (
    split_cases,
    fit_target_transform, apply_target_transform,
    fit_feature_scaler,   apply_feature_scaler,
)
from src.dataset        import TrajectoryDataset
from src.model          import build_model
from src.train          import train
from src.evaluate       import predict_dataset, compute_metrics
from src.visualize      import (
    plot_training_history,
    plot_case_trajectories,
    parity_plot,
    residual_plot,
)

print(f"Device: {device}\n")


# ── 1. Load data ──────────────────────────────────────────────────────────────
all_cases, kept_cases, removed_cases, report_df = load_all_cases()

print("=== LOAD REPORT ===")
print(report_df.to_string(index=False))

print("\nKept cases   :", [int(df["case_id"].iloc[0]) for df in kept_cases])
print("Removed cases:", [int(df["case_id"].iloc[0]) for df in removed_cases])

if removed_cases:
    print("\nRemoved spiky files:")
    for df in removed_cases:
        print(f"  Case {int(df['case_id'].iloc[0])}: {df['source_file'].iloc[0]}")


# ── 2. Split ──────────────────────────────────────────────────────────────────
train_cases, val_cases, test_cases, train_ids, val_ids, test_ids = split_cases(kept_cases)

print(f"\nTrain: {train_ids}")
print(f"Val  : {val_ids}")
print(f"Test : {test_ids}")


# ── 3. Transform target (fit on train only) ───────────────────────────────────
y_mean, y_stddev = fit_target_transform(train_cases)
train_cases = apply_target_transform(train_cases, y_mean, y_stddev)
val_cases   = apply_target_transform(val_cases,   y_mean, y_stddev)
test_cases  = apply_target_transform(test_cases,  y_mean, y_stddev)


# ── 4. Scale features (fit on train only) ────────────────────────────────────
scaler      = fit_feature_scaler(train_cases)
train_cases = apply_feature_scaler(train_cases, scaler)
val_cases   = apply_feature_scaler(val_cases,   scaler)
test_cases  = apply_feature_scaler(test_cases,  scaler)


# ── 5. Build datasets ─────────────────────────────────────────────────────────
train_dataset = TrajectoryDataset(train_cases)
val_dataset   = TrajectoryDataset(val_cases)
test_dataset  = TrajectoryDataset(test_cases)


# ── 6. Train ──────────────────────────────────────────────────────────────────
model   = build_model()
model, history = train(model, train_dataset, val_dataset)


# ── 7. Training history plot ──────────────────────────────────────────────────
plot_training_history(history)


# ── 8. Predict ────────────────────────────────────────────────────────────────
train_results = predict_dataset(model, train_dataset, y_mean, y_stddev)
val_results   = predict_dataset(model, val_dataset,   y_mean, y_stddev)
test_results  = predict_dataset(model, test_dataset,  y_mean, y_stddev)


# ── 9. Metrics ────────────────────────────────────────────────────────────────
compute_metrics(train_results, "TRAIN")
compute_metrics(val_results,   "VAL")
compute_metrics(test_results,  "TEST")


# ── 10. Plots ─────────────────────────────────────────────────────────────────
print("\n=== VALIDATION TRAJECTORIES ===")
plot_case_trajectories(val_results,  title_prefix="Validation case")

print("\n=== TEST TRAJECTORIES ===")
plot_case_trajectories(test_results, title_prefix="Test case")

parity_plot(test_results,   title="Test set — parity plot")
residual_plot(test_results, title="Test set — residual plot")


# ── 11. Save outputs ──────────────────────────────────────────────────────────
report_df.to_csv(os.path.join(OUTPUT_DIR, "case_spike_report.csv"), index=False)

if not train_results.empty:
    train_results.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)
if not val_results.empty:
    val_results.to_csv(  os.path.join(OUTPUT_DIR, "val_predictions.csv"),   index=False)
if not test_results.empty:
    test_results.to_csv( os.path.join(OUTPUT_DIR, "test_predictions.csv"),  index=False)

pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

print("\n=== SAVED TO outputs/ ===")
print("  case_spike_report.csv")
print("  training_history.csv")
print("  train/val/test predictions CSVs")
print("  models/best_node_model.pth")
