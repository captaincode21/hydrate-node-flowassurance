import os
import pandas as pd

from config import (
    DATA_DIR, RENAME_DICT, SELECTED_COLS, TARGET_COL,
    AUTO_REMOVE_SPIKY_CASES, SPIKE_ABS_THRESHOLD, SPIKE_RATIO_THRESHOLD,
    APPLY_TRIMMING,
)
from src.utils import detect_spiky_case, trim_steady_tail


def safe_read_table(file_path):
    """
    Read a tab-separated OLGA output file.
    Tries skiprows=1 first (common OLGA header format);
    falls back to no skip if the expected columns are missing.
    """
    try:
        df = pd.read_csv(file_path, sep="\t", engine="python", skiprows=1)
        if not all(col in df.columns for col in SELECTED_COLS):
            df = pd.read_csv(file_path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(file_path, sep="\t", engine="python")
    return df


def load_all_cases():
    """
    Load every .txt / .csv file from DATA_DIR.
    Returns:
        all_cases   : list of DataFrames (all files successfully loaded)
        kept_cases  : all_cases minus spiky ones (if AUTO_REMOVE_SPIKY_CASES)
        removed_cases: spiky DataFrames
        report_df   : summary DataFrame with spike stats per case
    """
    all_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".txt", ".csv"))
    ])

    all_cases   = []
    load_report = []

    for i, file_name in enumerate(all_files, start=1):
        file_path = os.path.join(DATA_DIR, file_name)
        df = safe_read_table(file_path)

        missing = [c for c in SELECTED_COLS if c not in df.columns]
        if missing:
            print(f"[SKIP] {file_name} — missing columns: {missing}")
            continue

        df = df[SELECTED_COLS].copy()
        df = df.rename(columns=RENAME_DICT)

        # Basic cleaning
        df = df.dropna(subset=["time_s", TARGET_COL]).copy()
        df = (df.sort_values("time_s")
                .drop_duplicates(subset=["time_s"])
                .reset_index(drop=True))
        df = df[df["time_s"].diff().fillna(1) >= 0].reset_index(drop=True)

        if APPLY_TRIMMING:
            df = trim_steady_tail(df, TARGET_COL)

        df["case_id"]     = i
        df["source_file"] = file_name

        is_spiky, stats = detect_spiky_case(
            df, TARGET_COL,
            abs_thresh=SPIKE_ABS_THRESHOLD,
            ratio_thresh=SPIKE_RATIO_THRESHOLD,
        )
        df["is_spiky_case"] = is_spiky

        load_report.append({
            "case_id":               i,
            "source_file":           file_name,
            "rows":                  len(df),
            "target_max":            stats["max"],
            "target_median_positive": stats["median_pos"],
            "spike_ratio":           stats["ratio"],
            "is_spiky":              is_spiky,
        })

        all_cases.append(df)

    report_df = pd.DataFrame(load_report)

    if AUTO_REMOVE_SPIKY_CASES:
        kept_cases    = [df for df in all_cases if not bool(df["is_spiky_case"].iloc[0])]
        removed_cases = [df for df in all_cases if     bool(df["is_spiky_case"].iloc[0])]
    else:
        kept_cases    = all_cases[:]
        removed_cases = []

    return all_cases, kept_cases, removed_cases, report_df
