# src/myrrr/mixins.py
import torch
import pandas as pd
import numpy as np

__all__ = ["UtilityMixin", "PlottingSamplingMixin"]

class UtilityMixin:
    """
    Shared utilities for estimators:
    - _to_tensor: device-aware conversion
    - _ridge_ls: multiresponse ridge via SVD
    """

    def _to_tensor(self, x, dtype=None, device=None):
        """
        Convert array-like to torch.Tensor and move to device.
        If device is None, default to self.device if present.
        """
        dev = device if device is not None else getattr(self, "device", None)
        if isinstance(x, torch.Tensor):
            t = x
            # Allow dtype override
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
        else:
            t = torch.as_tensor(x, dtype=(dtype or torch.float32))
        return t.to(dev) if dev is not None else t

    @staticmethod
    @torch.no_grad()
    def _ridge_ls(X, Y, lam):
        """
        Multiresponse ridge via SVD:
          X = U S V^T, C = V diag(S / (S^2 + lam)) U^T Y
        X: (n x p), Y: (n x q), lam: scalar >= 0
        Returns C: (p x q) torch.Tensor
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)     # U: (n x r), S: (r,), Vh: (r x p)
        UtY = U.transpose(-1, -2) @ Y                           # (r x q)
        shrink = S / (S.square() + lam)                         # (r,)
        C = Vh.transpose(-1, -2) @ (shrink.unsqueeze(1) * UtY)  # (p x q)
        return C
    

class PlottingSamplingMixin:
    """
    Shared plotting utilities for sampling-based estimators.
    """

    # -- Stack one metric across bins into a DataFrame (rows=bins, cols=parts)
    def stack_metric(self, metrics_by_bin, key):
        rows = {}
        for b, d in metrics_by_bin.items():
            s = d[key]
            # Accept Series or 1D np.ndarray; convert to Series if needed
            if isinstance(s, np.ndarray):
                s = pd.Series(s, index=d.get("Y_test_comp_df").columns)
            rows[b] = s
        return pd.DataFrame(rows).T  # bins as rows
    
    
    # -- Build all four metric DataFrames at once
    def build_metric_frames(self, metrics_by_bin):
        return {
            "cors_comp": self.stack_metric(metrics_by_bin, "cors_comp"),
            "cors_clr_comp": self.stack_metric(metrics_by_bin, "cors_clr_comp"),
            "Rsq_comp": self.stack_metric(metrics_by_bin, "Rsq_comp"),
            "Rsq_clr_comp": self.stack_metric(metrics_by_bin, "Rsq_clr_comp"),
        }