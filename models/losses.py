import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from scipy.stats import gaussian_kde


class KDEWeightedHuberLoss(nn.Module):
    def __init__(self, kde_reference_csv=None, eps=1e-6, delta=5.0, delta_source="default"):
        super().__init__()
        self.eps = eps
        self.delta = float(delta)
        self.delta_source = str(delta_source)
        self.kde_model = None
        self.min_density = 0.0
        self.max_density = 1.0

        print(f"[LossConfig] delta={self.delta:.6g} source={self.delta_source}")

        if kde_reference_csv and os.path.exists(kde_reference_csv):
            y_train = pd.read_csv(kde_reference_csv).to_numpy().flatten()
            if y_train.size >= 2:
                kde = gaussian_kde(y_train)
                dens = kde(y_train)
                self.kde_model = kde
                self.min_density = float(dens.min())
                self.max_density = float(dens.max())
                kde_cap = float(np.mean(y_train))
                if kde_cap < self.delta:
                    warnings.warn(
                        (
                            f"[LossConfig] delta override detected: base={self.delta:.6g} "
                            f"(source={self.delta_source}) -> effective={kde_cap:.6g} "
                            "due to KDE reference mean cap."
                        ),
                        UserWarning,
                    )
                    self.delta = kde_cap

        print(f"[LossConfig] effective delta={self.delta:.6g} source={self.delta_source}")

    def _get_kde_weight(self, y_vals):
        if self.kde_model is None:
            return np.ones((y_vals.shape[0],), dtype=np.float32)
        dens = self.kde_model(y_vals[:, 0])
        dens = (dens - self.min_density) / (self.max_density - self.min_density + self.eps)
        dens = 1.0 - dens
        dens = np.maximum(dens, self.eps)
        return (dens / np.mean(dens)).astype(np.float32)

    def forward(self, pred, target):
        target_np = target.detach().cpu().numpy()
        weights_np = self._get_kde_weight(target_np)
        weights = torch.tensor(weights_np, dtype=torch.float32, device=pred.device).view(-1, 1)

        diff = pred - target
        abs_diff = torch.abs(diff)
        mse_loss = 0.5 * (diff ** 2)
        mae_loss = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, mse_loss, torch.abs(mae_loss))
        return torch.mean(loss * weights)

