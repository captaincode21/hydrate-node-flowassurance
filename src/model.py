import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from config import (
    device, FEATURE_COLS, HIDDEN_DIM,
    ODE_METHOD, ODE_RTOL, ODE_ATOL,
)


class FeatureInterpolator:
    """
    Linear interpolation of feature matrix x over time grid t.
    Used to query continuous feature values at any ODE solver time step.
    """

    def __init__(self, t_grid: torch.Tensor, x_grid: torch.Tensor):
        self.t_grid = t_grid
        self.x_grid = x_grid

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        tg, xg = self.t_grid, self.x_grid

        if t <= tg[0]:
            return xg[0]
        if t >= tg[-1]:
            return xg[-1]

        idx     = torch.searchsorted(tg, t).item()
        t0, t1  = tg[idx - 1], tg[idx]
        x0, x1  = xg[idx - 1], xg[idx]
        w       = (t - t0) / (t1 - t0 + 1e-8)
        return x0 + w * (x1 - x0)


class NODEFunc(nn.Module):
    """
    Neural ODE dynamics function.

    dy/dt = f(y, u(t))  where u(t) is the interpolated feature vector.

    Architecture: [y | u(t)] → Linear → Tanh → Linear → Tanh → Linear → dy/dt
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_interp = None

    def set_interpolator(self, interp: FeatureInterpolator):
        self.feature_interp = interp

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_t  = self.feature_interp(t).view(1, -1)
        y_in = y.view(1, 1)
        inp  = torch.cat([y_in, u_t], dim=1)
        return self.net(inp).view_as(y)


def build_model() -> NODEFunc:
    """Instantiate and return the NODE model on the configured device."""
    return NODEFunc(input_dim=len(FEATURE_COLS), hidden_dim=HIDDEN_DIM).to(device)


def run_trajectory(
    model: NODEFunc,
    t_np,
    x_np,
    y0_np,
) -> torch.Tensor:
    """
    Integrate the NODE over a single trajectory.

    Args:
        model  : NODEFunc instance
        t_np   : [T] numpy array of timestamps
        x_np   : [T, F] numpy array of features
        y0_np  : [1] numpy array — initial condition

    Returns:
        pred   : [T] torch tensor of predicted target values
    """
    t   = torch.tensor(t_np,   dtype=torch.float32, device=device)
    x   = torch.tensor(x_np,   dtype=torch.float32, device=device)
    y0  = torch.tensor(y0_np,  dtype=torch.float32, device=device)

    model.set_interpolator(FeatureInterpolator(t, x))

    pred = odeint(
        model, y0, t,
        method=ODE_METHOD,
        rtol=ODE_RTOL,
        atol=ODE_ATOL,
    )
    return pred.squeeze(-1)
