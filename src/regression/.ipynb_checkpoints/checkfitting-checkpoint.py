import pandas as pd
import numpy as np

__all__ = ["CheckFitting"]

class CheckFitting:
    def __init__(self, test_index, col_names):
        self.test_index = test_index
        self.col_names = col_names
    def convert_df(self, Y):
        return pd.DataFrame(Y, index = self.test_index, columns = self.col_names)
    
    def corr(self, Y_pred, Y_obs):
        Y_pred_df = self.convert_df(Y_pred)
        Y_obs_df = self.convert_df(Y_obs)
        return Y_pred_df.corrwith(Y_obs_df, axis = 0)
    def rsquared_feature(self, Y_pred, Y_obs):
        SSE = np.sum((Y_pred - Y_obs)**2, axis = 0)
        SST = np.sum((Y_obs - np.mean(Y_obs, axis = 0))**2, axis = 0)
        return 1 - SSE / SST
    def rsquared(self, Y_pred, Y_obs):
        SSE = np.sum((Y_pred - Y_obs)**2)
        SST = np.sum((Y_obs - np.mean(Y_obs))**2)
        return 1 - SSE / SST
    def mse(self, Y_pred, Y_obs, sample_weight=None):
        """
        Mean squared error on the held-out test set.
        If sample_weight is provided (length n_test), computes weighted MSE:
            MSE_w = [sum_i w_i sum_j (yhat_ij - y_ij)^2] / [sum_i w_i]
        Otherwise returns the unweighted mean of squared residuals.
        """
        resid2 = (Y_pred - Y_obs) ** 2
        if sample_weight is None:
            return float(np.mean(resid2))
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1, 1)  # (n_test x 1)
            if w.shape[0] != resid2.shape[0]:
                raise ValueError(f"sample_weight length {w.shape[0]} != n_test {resid2.shape[0]}")
            w = np.clip(w, 0.0, None)
            SSE_w = float((w * resid2).sum())
            w_sum = float(w.sum())
            return SSE_w / w_sum if w_sum > 0 else np.nan
        