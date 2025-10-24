# src/myrrr/estimators.py
import numpy as np
import torch
from .mixins import UtilityMixin

class RidgeRRR(UtilityMixin):
    """
    Ridge-regularized Reduced-Rank Regression (RRR) with optional response weighting.
    """

    def __init__(self, rank=1, lam=0.0, sv_tol=1e-12, device=None, response_weight=None):
        self.rank = int(rank)
        self.lam = float(lam)
        self.sv_tol = float(sv_tol)
        self.device = device
        self.response_weight = response_weight

        # Fitted attributes
        self.coef_ridge_ = None   # (p x q)
        self.coef_ = None         # (p x q)
        self.V_ = None            # (q x r_eff)
        self.svals_ = None        # (r_eff,)
        self.rank_eff_ = 0
        self.is_fitted_ = False

    def _build_weight(self, q, dtype):
        """Compute sqrt(G) and sqrtinv(G) if response_weight is set."""
        if self.response_weight is None:
            return None, None
        G = self._to_tensor(self.response_weight, dtype=dtype)
        evals, Q = torch.linalg.eigh(G)
        evals = torch.clamp(evals, min=0.0)
        sqrtG = Q @ (torch.sqrt(evals).unsqueeze(0) * Q.transpose(-1, -2))
        sqrtinvG = Q @ ((1.0 / torch.sqrt(torch.clamp(evals, min=1e-12))).unsqueeze(0) *
                        Q.transpose(-1, -2))
        return sqrtG, sqrtinvG

    @torch.no_grad()
    def fit(self, X, Y):
        X = self._to_tensor(X)
        Y = self._to_tensor(Y, dtype=X.dtype)
        n, p = X.shape
        _, q = Y.shape

        # 1) Ridge baseline
        C_lam = self._ridge_ls(X, Y, lam=self.lam)      # (p x q)

        # 2) Weighted/unweighted fitted responses
        sqrtG, sqrtinvG = self._build_weight(q, dtype=Y.dtype)
        XC = X @ C_lam if sqrtG is None else X @ C_lam @ sqrtG

        # 3) Truncated SVD
        U, S, Vh = torch.linalg.svd(XC, full_matrices=False)
        rmax = Vh.shape[0]
        if self.rank <= 0 or rmax == 0:
            coef_rr = torch.zeros((p, q), dtype=X.dtype, device=X.device)
            self.coef_ridge_ = C_lam
            self.coef_ = coef_rr
            self.V_ = None
            self.svals_ = None
            self.rank_eff_ = 0
            self.is_fitted_ = True
            return self

        r_eff = min(self.rank, rmax)
        mask = S[:r_eff] > self.sv_tol
        r_eff = int(mask.sum().item())
        if r_eff == 0:
            coef_rr = torch.zeros((p, q), dtype=X.dtype, device=X.device)
            self.coef_ridge_ = C_lam
            self.coef_ = coef_rr
            self.V_ = None
            self.svals_ = None
            self.rank_eff_ = 0
            self.is_fitted_ = True
            return self

        V = Vh.transpose(-1, -2)    # (q x rmax)
        Vr = V[:, :r_eff]            # (q x r_eff)
        P = Vr @ Vr.transpose(-1, -2)

        # 4) Reduced-rank coefficients
        coef_rr = C_lam @ P if sqrtG is None else C_lam @ sqrtG @ P @ sqrtinvG

        # Store
        self.coef_ridge_ = C_lam
        self.coef_ = coef_rr
        self.V_ = Vr
        self.svals_ = S[:r_eff]
        self.rank_eff_ = r_eff
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Estimator not fitted. Call fit(X, Y) first.")
        Xnew = self._to_tensor(X, dtype=self.coef_.dtype)
        Yhat = Xnew @ self.coef_
        return Yhat.detach().cpu().numpy()

    def get_params(self, deep=True):
        return dict(rank=self.rank, lam=self.lam, sv_tol=self.sv_tol,
                    device=self.device, response_weight=self.response_weight)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class RidgeRRRCV(UtilityMixin):
    """
    Cross-validation over rank and lambda for RidgeRRR.
    """

    def __init__(self, rank_grid=None, lambda_grid=None, n_splits=5,
                 seed=0, shuffle=True, sv_tol=1e-12, device=None,
                 response_weight=None):
        self.rank_grid = rank_grid
        self.lambda_grid = lambda_grid
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle
        self.sv_tol = sv_tol
        self.device = device
        self.response_weight = response_weight

        self.cv_errors_ = None
        self.best_rank_ = None
        self.best_lambda_ = None
        self.best_estimator_ = None

    @torch.no_grad()
    def fit(self, X, Y):
        from sklearn.model_selection import KFold

        X = self._to_tensor(X)
        Y = self._to_tensor(Y, dtype=X.dtype)
        n, p = X.shape
        _, q = Y.shape

        # Default grids
        if self.rank_grid is None:
            self.rank_grid = list(range(0, min(p, q) + 1))
        if self.lambda_grid is None:
            self.lambda_grid = (10.0 ** torch.linspace(-6, 4, steps=12)).tolist()

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        cr_path = torch.empty((len(self.rank_grid), len(self.lambda_grid), self.n_splits),
                              dtype=X.dtype, device=X.device)

        # Precompute response weight transforms if any
        if self.response_weight is not None:
            G = self._to_tensor(self.response_weight, dtype=Y.dtype)
            evals, Q = torch.linalg.eigh(G)
            evals = torch.clamp(evals, min=0.0)
            sqrtG = Q @ (torch.sqrt(evals).unsqueeze(0) * Q.transpose(-1, -2))
            sqrtinvG = Q @ ((1.0 / torch.sqrt(torch.clamp(evals, min=1e-12))).unsqueeze(0) *
                            Q.transpose(-1, -2))
        else:
            sqrtG = None
            sqrtinvG = None

        # CV loop
        for fold_id, (tr_idx_np, va_idx_np) in enumerate(kf.split(np.arange(n))):
            tr_idx = torch.as_tensor(tr_idx_np, device=X.device)
            va_idx = torch.as_tensor(va_idx_np, device=X.device)
            Xtr, Ytr = X[tr_idx], Y[tr_idx]
            Xva, Yva = X[va_idx], Y[va_idx]

            for li, lam in enumerate(self.lambda_grid):
                # Ridge LS via mixin staticmethod
                C_lam = self._ridge_ls(Xtr, Ytr, lam=float(lam))
                Yhat_ls_va = Xva @ C_lam

                # Responses for SVD
                XC_tr = Xtr @ C_lam if sqrtG is None else Xtr @ C_lam @ sqrtG

                # SVD
                U, S, Vh = torch.linalg.svd(XC_tr, full_matrices=False)
                V = Vh.transpose(-1, -2)  # (q x rmax)

                # Rank 0 baseline: predict zeros (centered setting)
                cr_path[0, li, fold_id] = torch.sum((Yva - 0.0) ** 2)

                for ri, r in enumerate(self.rank_grid[1:], start=1):
                    r_eff = min(r, V.shape[1])
                    if r_eff == 0 or (S[:r_eff] <= self.sv_tol).all():
                        Yhat_rr_va = torch.zeros_like(Yva)
                    else:
                        Vr = V[:, :r_eff]
                        P = Vr @ Vr.transpose(-1, -2)
                        if sqrtG is None:
                            Yhat_rr_va = Yhat_ls_va @ P
                        else:
                            Yhat_rr_va = (Yhat_ls_va @ sqrtG) @ P @ sqrtinvG
                    cr_path[ri, li, fold_id] = torch.sum((Yva - Yhat_rr_va) ** 2)

        # Aggregate
        cr_error = cr_path.mean(dim=2)
        idx = torch.argmin(cr_error)
        ri, li = np.unravel_index(int(idx), cr_error.shape)
        self.best_rank_ = int(self.rank_grid[ri])
        self.best_lambda_ = float(self.lambda_grid[li])
        self.cv_errors_ = cr_error.detach().cpu().numpy()

        # Fit final estimator
        est = RidgeRRR(rank=self.best_rank_, lam=self.best_lambda_,
                       sv_tol=self.sv_tol, device=self.device,
                       response_weight=self.response_weight)
        est.fit(X, Y)
        self.best_estimator_ = est
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Cross-validator not fitted. Call fit(X, Y) first.")
        return self.best_estimator_.predict(X)

    def get_params(self, deep=True):
        return dict(rank_grid=self.rank_grid, lambda_grid=self.lambda_grid,
                    n_splits=self.n_splits, seed=self.seed, shuffle=self.shuffle,
                    sv_tol=self.sv_tol, device=self.device,
                    response_weight=self.response_weight)