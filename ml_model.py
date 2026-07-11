"""
ml_model.py
-----------
A simple multilayer CNN that emulates the EnKF analysis, and a thin wrapper that
lets it drive `Experiment` in place of the EnKF.

Input  (2 channels): x_f            -- ensemble-mean forecast
                     y - x_f        -- ensemble-mean innovation (obs minus forecast)
Output (1 channel):  x_a            -- ensemble-mean analysis

The network predicts the analysis *increment* (x_a - x_f) from the two input
channels; the forecast is added back on reconstruction
(x_a = x_f + CNN([x_f, y - x_f])). Every convolution uses circular padding so the
doubly-periodic boundary conditions are respected.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:                       # torch optional at import time
    torch = None
    nn = object


class PeriodicCNN(nn.Module):
    """Multilayer CNN with circular padding; predicts the analysis increment."""

    def __init__(self, in_ch=2, width=32, depth=4):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1, padding_mode="circular"),
                       nn.ReLU()]
            c = width
        layers += [nn.Conv2d(c, 1, 3, padding=1, padding_mode="circular")]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                # x: (B, 2, N, N) normalized
        return self.net(x)               # analysis increment (x_a - x_f), normalized


class CNNAssimilator:
    """
    Wraps a trained PeriodicCNN so it exposes `assimilate(xf_mean, y, obs_idx)`
    (the interface Experiment expects). Fields are normalized by a scalar sigma.
    """

    def __init__(self, model, sigma, N, device="cpu"):
        self.model = model.to(device).eval()
        self.sigma = float(sigma)
        self.N = int(N)
        self.device = device

    @classmethod
    def load(cls, path, N, device="cpu", **kw):
        ckpt = torch.load(path, map_location=device)
        model = PeriodicCNN(**{**ckpt.get("arch", {}), **kw})
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["sigma"], N, device=device)

    def assimilate(self, xf_mean, y, obs_idx):
        """Return the analysis mean (flat, length N*N) from forecast + obs."""
        N = self.N
        xf = xf_mean.reshape(N, N)
        innov = np.zeros(N * N, dtype=np.float64)
        innov[obs_idx] = y - xf_mean[obs_idx]      # 0 where unobserved
        inp = np.stack([xf, innov.reshape(N, N)]) / self.sigma   # (2, N, N)
        with torch.no_grad():
            t = torch.tensor(inp[None], dtype=torch.float32, device=self.device)
            incr = self.model(t)[0, 0].cpu().numpy()             # increment (normalized)
        return (xf + incr * self.sigma).ravel()                  # x_a = x_f + increment


def get_device(verbose=True):
    """Return a torch device, reporting GPU availability.

    Use a GPU when present (local workstation or Colab GPU runtime); otherwise
    fall back to CPU. Training on CPU is slow -- run notebook 04 on a GPU
    (e.g. Google Colab) and keep inference local.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if dev.type == "cuda":
            print("GPU available:", torch.cuda.get_device_name(0))
        else:
            print("No GPU found -- using CPU. Training will be slow; consider "
                  "running this notebook on Google Colab (GPU runtime).")
    return dev


def build_dataset(run, obs, sigma=None):
    """
    Build (inputs, targets) from a stored EnKF run and its observations.

    :param run: Dataset with 'xf_mean', 'xa_mean' (dims time, y, x)
    :param obs: Dataset with 'obs' (dims time, y, x); NaN where unobserved
    :return: X (T, 2, N, N), Y (T, 1, N, N), sigma
             X channels = [x_f, y - x_f];  Y = x_a - x_f (analysis increment).
             Cycle 0 is skipped.
    """
    xf = run["xf_mean"].values[1:]                 # skip initial cycle
    xa = run["xa_mean"].values[1:]
    yy = np.nan_to_num(obs["obs"].values[1:len(xf) + 1])
    innov = yy - xf
    if sigma is None:
        sigma = float(np.std(xf))
    X = np.stack([xf, innov], axis=1) / sigma      # (T, 2, N, N)
    Y = ((xa - xf) / sigma)[:, None]                # (T, 1, N, N)  analysis increment
    return X.astype(np.float32), Y.astype(np.float32), sigma
