import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_FEATURES


class VolGRU(nn.Module):
    """GRU over a 30-day feature window -> next-5-day annualized volatility."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return F.softplus(self.head(out[:, -1])).squeeze(-1)  # volatility is always positive


class Predictor:
    def __init__(self, path: str, allow_untrained: bool = False):
        self.net = VolGRU().eval()
        self.mean, self.std = torch.zeros(N_FEATURES), torch.ones(N_FEATURES)
        self.info = {"trained": False}
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu")
            self.net.load_state_dict(ck["state_dict"])
            self.mean, self.std = torch.tensor(ck["mean"]), torch.tensor(ck["std"])
            self.info = {"trained": True, **ck.get("metrics", {})}
        elif not allow_untrained:
            raise FileNotFoundError(f"{path} not found. Run `python -m training.train` first "
                                    "(or set ALLOW_UNTRAINED=1 for a UI-only demo).")

    @torch.inference_mode()
    def __call__(self, batch: np.ndarray) -> np.ndarray:
        x = (torch.from_numpy(batch) - self.mean) / self.std
        return self.net(x).numpy()
