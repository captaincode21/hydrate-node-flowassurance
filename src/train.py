import os
import numpy as np
import torch
import torch.nn as nn

from config import (
    device, MODEL_DIR,
    LR, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, GRAD_CLIP,
    USE_DERIVATIVE_REG, DERIV_LAMBDA,
)
from src.model import run_trajectory


criterion = nn.MSELoss()


def derivative_loss(pred: torch.Tensor, true: torch.Tensor, t_np) -> torch.Tensor:
    """Penalise mismatch in finite-difference derivatives of pred vs true."""
    if len(t_np) < 3:
        return torch.tensor(0.0, device=device)

    t  = torch.tensor(t_np, dtype=torch.float32, device=device)
    dt = torch.clamp(t[1:] - t[:-1], min=1e-8)

    dp       = (pred[1:] - pred[:-1]) / dt
    dt_true  = (true[1:] - true[:-1]) / dt
    return torch.mean((dp - dt_true) ** 2)


def evaluate_dataset(model, dataset) -> float:
    """Return mean MSE loss over all trajectories in a dataset (no grad)."""
    model.eval()
    losses = []

    with torch.no_grad():
        for t_np, x_np, y_np, cid, _ in dataset:
            pred = run_trajectory(model, t_np, x_np, y_np[0])
            true = torch.tensor(y_np.squeeze(-1), dtype=torch.float32, device=device)
            losses.append(criterion(pred, true).item())

    return float(np.mean(losses)) if losses else float("nan")


def train(model, train_dataset, val_dataset):
    """
    Full training loop with early stopping.

    Returns:
        model    : best-checkpoint model (weights already loaded)
        history  : dict with keys 'epoch', 'train_loss', 'val_loss'
    """
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val    = float("inf")
    best_state  = None
    best_epoch  = -1
    wait        = 0
    history     = {"epoch": [], "train_loss": [], "val_loss": []}
    model_path  = os.path.join(MODEL_DIR, "best_node_model.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for t_np, x_np, y_np, cid, _ in train_dataset:
            optimizer.zero_grad()

            pred = run_trajectory(model, t_np, x_np, y_np[0])
            true = torch.tensor(y_np.squeeze(-1), dtype=torch.float32, device=device)

            loss = criterion(pred, true)
            if USE_DERIVATIVE_REG:
                loss = loss + DERIV_LAMBDA * derivative_loss(pred, true, t_np)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss   = evaluate_dataset(model, val_dataset)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest epoch : {best_epoch}")
    print(f"Best val   : {best_val:.6f}")
    print(f"Model saved: {model_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    return model, history
