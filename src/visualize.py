import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history: dict):
    """Plot train vs validation loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"],   label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_case_trajectories(df_results: pd.DataFrame, title_prefix: str = "Case"):
    """Plot true vs predicted trajectory for every case_id in the results."""
    if df_results.empty:
        return

    for cid in sorted(df_results["case_id"].unique()):
        dfi = df_results[df_results["case_id"] == cid].copy()

        plt.figure(figsize=(9, 5))
        plt.plot(dfi["time_s"], dfi["true_target"],  label="True",      linewidth=2)
        plt.plot(dfi["time_s"], dfi["pred_target"],  label="Predicted", linewidth=2, linestyle="--")
        plt.xlabel("Time [s]")
        plt.ylabel("Hydrate mass in oil at topside [kg/m³]")
        plt.title(f"{title_prefix} {cid}: True vs Predicted")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def parity_plot(df_results: pd.DataFrame, title: str = "Parity plot"):
    """Scatter plot of observed vs predicted values with a 1:1 reference line."""
    if df_results.empty:
        return

    mn = min(df_results["true_target"].min(), df_results["pred_target"].min())
    mx = max(df_results["true_target"].max(), df_results["pred_target"].max())

    plt.figure(figsize=(6, 6))
    plt.scatter(df_results["true_target"], df_results["pred_target"], alpha=0.6)
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    plt.xlabel("Observed hydrate mass [kg/m³]")
    plt.ylabel("Predicted hydrate mass [kg/m³]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def residual_plot(df_results: pd.DataFrame, title: str = "Residual plot"):
    """Plot residuals (true − predicted) against predicted values."""
    if df_results.empty:
        return

    df = df_results.copy()
    df["residual"] = df["true_target"] - df["pred_target"]

    plt.figure(figsize=(8, 5))
    plt.scatter(df["pred_target"], df["residual"], alpha=0.6)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Predicted hydrate mass [kg/m³]")
    plt.ylabel("Residual (true − predicted)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
