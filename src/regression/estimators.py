# src/myrrr/estimators.py
import numpy as np
import torch
from .mixins import UtilityMixin
from .checkfitting import CheckFitting
from .preprocessing import StandardScalerX, CenterY
from skbio.stats.composition import clr






__all__ = ["RidgeRRR", "RidgeRRRCV", "RRRBinEvaluator"]

class RidgeRRR(UtilityMixin):
    """
    Ridge-regularized Reduced-Rank Regression (RRR) with optional response weighting.
    """

    def __init__(self, rank=1, lam=0.0, sv_tol=1e-12, device=None, sample_weight = None, response_weight=None):
        self.rank = int(rank)
        self.lam = float(lam)
        self.sv_tol = float(sv_tol)
        self.device = device
        self.response_weight = response_weight
        self.sample_weight = sample_weight

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

        # Apply sample weights if any
        # (weights are sqrt-scaled for both X and Y)
        if self.sample_weight is not None:
            w = self._to_tensor(self.sample_weight, dtype=X.dtype).reshape(-1)
            if w.shape[0] != n:
                raise ValueError(f"sample_weight length {w.shape[0]} does not match n={n}")
            wsqrt = torch.sqrt(torch.clamp(w, min=0.0)).view(n, 1)  # (n x 1)
            X = X * wsqrt
            Y = Y * wsqrt

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
        self.projector = P if sqrtG is None else sqrtG @ P @ sqrtinvG
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
                    device=self.device, response_weight=self.response_weight,
                    sample_weight=self.sample_weight)

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
                 sample_weight=None,
                 response_weight=None):
        self.rank_grid = rank_grid
        self.lambda_grid = lambda_grid
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle
        self.sv_tol = sv_tol
        self.device = device
        self.response_weight = response_weight
        self.sample_weight = sample_weight
        self.cr_path = None

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

        if self.sample_weight is not None:
            w_all = self._to_tensor(self.sample_weight, dtype=X.dtype).reshape(-1)
            if w_all.shape[0] != n:
                raise ValueError(f"sample_weight length {w_all.shape[0]} does not match n={n}")
        else:
            w_all = None

        # Default grids
        if self.rank_grid is None:
            self.rank_grid = list(range(1, min(p, q) + 1))
        if self.lambda_grid is None:
            self.lambda_grid = (10.0 ** torch.linspace(-6, 4, steps=12)).tolist()

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        # cr_path = torch.empty((len(self.rank_grid), len(self.lambda_grid), self.n_splits),
        #                       dtype=X.dtype, device=X.device)
        cr_path = torch.full((len(self.rank_grid), len(self.lambda_grid), self.n_splits), float('inf'), dtype=X.dtype, device=X.device)

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

            if w_all is not None:
                w_tr = w_all[tr_idx]
                wsqrt_tr = torch.sqrt(torch.clamp(w_tr, min=0.0)).view(-1, 1)
                Xtr_w = Xtr * wsqrt_tr
                Ytr_w = Ytr * wsqrt_tr
                # VALIDATION: keep unweighted Xva, Yva; use w_va to weight residuals
                w_va = w_all[va_idx].view(-1, 1)  # (n_va x 1)

            else:
                Xtr_w, Ytr_w = Xtr, Ytr
                w_va = None

            for li, lam in enumerate(self.lambda_grid):
                # Ridge LS via mixin staticmethod
                # Ridge LS on row-weighted training data

                C_lam = self._ridge_ls(Xtr_w, Ytr_w, lam=float(lam))
                Yhat_ls_va = Xva @ C_lam

                # Responses for SVD
                XC_tr = Xtr_w @ C_lam if sqrtG is None else Xtr_w @ C_lam @ sqrtG

                # SVD
                U, S, Vh = torch.linalg.svd(XC_tr, full_matrices=False)
                V = Vh.transpose(-1, -2)  # (q x rmax)
                rmax = Vh.shape[0]
                count_above = int((S > self.sv_tol).sum().item())
                # print(f"[fold {fold_id}, lam={lam:.3g}] rmax={rmax}, count_above_tol={count_above}, top S={S[:5].cpu().numpy()}")
                
                # Rank 0 baseline: predict zeros (centered setting)
                # resid0 = Yva_w - 0.0
                # if w_va is not None:
                #     cr_path[0, li, fold_id] = torch.sum((resid0 ** 2) * w_va)
                # else:
                #     cr_path[0, li, fold_id] = torch.sum(resid0 ** 2)


                for ri, r in enumerate(self.rank_grid):
                    # r_eff = min(r, V.shape[1])
                    r_eff = min(r, count_above)
                    if r_eff == 0:
                        # Yhat_rr_va = torch.zeros_like(Yva)
                        cr_path[ri, li, fold_id] = float('inf')
                        continue
                    else:
                        Vr = V[:, :r_eff]
                        P = Vr @ Vr.transpose(-1, -2)
                        if sqrtG is None:
                            Yhat_rr_va = Yhat_ls_va @ P
                        else:
                            Yhat_rr_va = (Yhat_ls_va @ sqrtG) @ P @ sqrtinvG

                        resid = Yva - Yhat_rr_va
                        if w_va is not None:
                            cr_path[ri, li, fold_id] = torch.sum((resid ** 2) * w_va)
                        else:
                            cr_path[ri, li, fold_id] = torch.sum(resid ** 2)
        
        self.cr_path = cr_path
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
                       sample_weight=self.sample_weight,
                       response_weight=self.response_weight)
        est.fit(X, Y)
        self.best_estimator_ = est
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Cross-validator not fitted. Call fit(X, Y) first.")
        return self.best_estimator_.predict(X)

    def get_params(self):
        return dict(rank_grid=self.rank_grid, lambda_grid=self.lambda_grid,
                    n_splits=self.n_splits, seed=self.seed, shuffle=self.shuffle,
                    sv_tol=self.sv_tol, device=self.device,
                    sample_weight=self.sample_weight,
                    response_weight=self.response_weight,
                    cr_path = self.cr_path)
    


class RRRBinEvaluator:
    """
    Evaluate RidgeRRRCV across TrainTestSplit bins (folds).

    Parameters
    ----------
    device : torch.device or None
    rank_grid : list[int] or None
    lambda_grid : list[float] or None
    n_splits_cv : int
        Inner CV folds used by RidgeRRRCV on each training split.

    Methods
    -------
    evaluate(X_design, Y_ilr, train_test, ilr_tx, sample_index, part_names)
        Returns metrics_by_bin dict keyed by fold id.

    Notes
    -----
    Weighted test loss:
        SSE_w = ∑_i w_i ||y_i - ŷ_i||^2
        MSE_w = SSE_w / ∑_i w_i
    Weighted R^2 per part:
        R^2_w = 1 - SSE_w / SST_w, where SST_w = ∑_i w_i (y_i - μ_w)^2,
        μ_w = (∑_i w_i y_i) / (∑_i w_i)
    
    """

    def __init__(self, device=None, rank_grid=None, lambda_grid=None, n_splits_cv=5):
        self.device = device
        self.rank_grid = rank_grid
        self.lambda_grid = lambda_grid
        self.n_splits_cv = n_splits_cv

    def evaluate(self, X_design, Y_ilr, train_test, ilr_tx, cell_type_proportions_df, sample_weight=None, verbose = True):
        """
        Run RRR+CV for each bin, predict on held-out set, and compute metrics.

        Parameters
        ----------
        X_design : ndarray (n x p)
            Predictor matrix.
        Y_ilr : ndarray (n x q_ilr)
            ILR-transformed responses (centered later per split).
        train_test : TrainTestSplit
            Must provide split(X, Y, fold=...), filter_feature(X_train, X_test),
            and a .bins vector to identify the held-out indices.
        ilr_tx : CompositionalILR
            The transformer used earlier; must support inverse_transform(Y_ilr).

        cell_type_proportions_df : pandas.DataFrame (n x q)
            Original composition data frame, used to get sample indices and
            Column names of composition parts.
        sample_weight : array-like length n or None
            Row weights per sample (e.g., number of cells). If provided, used
            in inner CV and final fit to account for unequal sample sizes.

        Returns
        -------
        dict[int, dict]
            metrics_by_bin keyed by fold id.
        """
        from .estimators import RidgeRRRCV  # safe intra-module import

        metrics_by_bin = {}

        # Default grids if not provided
        # We’ll infer per-fold p,q after filtering/centering
        for fold_test in range(1, train_test.nfolds + 1):
            # Split into train/test for this fold
            X_train, X_test, Y_train, Y_test = train_test.split(X_design, Y_ilr, fold=fold_test)

            # Drop non-finite / zero-variance columns
            X_train, X_test = train_test.filter_feature(X_train, X_test)

            # Standardize X on training; center Y_ilr on training
            x_scaler = StandardScalerX()
            X_train_s = x_scaler.fit_transform(X_train)
            X_test_s = (X_test - x_scaler.mean_) / x_scaler.std_

            y_center = CenterY()
            Y_train_c = y_center.fit_transform(Y_train)

            if sample_weight is not None:
                # Build masks from bins to align with rows of X_design/Y_ilr
                test_mask = (train_test.bins == fold_test)
                train_mask = ~test_mask
                w_train = np.asarray(sample_weight)[train_mask]
                w_test = np.asarray(sample_weight)[test_mask]
            else:
                w_train = None
                w_test = None

            # Inner CV: pick rank and lambda using only training split
            p_eff, q_eff = X_train_s.shape[1], Y_train_c.shape[1]
            rank_grid = self.rank_grid or list(range(1, min(p_eff, q_eff) + 1))
            lambda_grid = self.lambda_grid or (10.0 ** np.linspace(-4, 4, 33)).tolist()

            cv = RidgeRRRCV(
                rank_grid=rank_grid,
                lambda_grid=lambda_grid,
                n_splits=self.n_splits_cv,
                device=self.device,
                sample_weight=w_train
            )
            cv.fit(X_train_s, Y_train_c)

            # Predict on held-out test set (centered ilr-space)
            Y_test_c_hat = cv.predict(X_test_s)
            # Undo centering (ilr-space)
            Y_test_ilr_hat = Y_test_c_hat + y_center.mean_
            # Invert ILR back to compositions
            Y_test_comp_hat = ilr_tx.inverse_transform(Y_test_ilr_hat)
            Y_true_obs_comp = ilr_tx.inverse_transform(Y_test)  # Y_test is ilr-space

            # Build index of held-out samples
            test_mask = (train_test.bins == fold_test)
            test_index = cell_type_proportions_df.index[test_mask]

            # Do fitting quality check
            checkfitting = CheckFitting(test_index = test_index, col_names = cell_type_proportions_df.columns)

            # Metrics in composition space
            cors_comp = checkfitting.corr(Y_pred = Y_test_comp_hat,
                                          Y_obs = Y_true_obs_comp)
            # Check R square in the composition space
            Rsq_comp = checkfitting.rsquared_feature(Y_pred = Y_test_comp_hat,
                                                     Y_obs = Y_true_obs_comp)
            # Check MSE in the composition space
            mse_comp = checkfitting.mse(Y_pred = Y_test_comp_hat,
                                        Y_obs = Y_true_obs_comp)

            # Metrics in CLR space
            Y_pred_clr = clr(Y_test_comp_hat)
            Y_obs_clr = clr(Y_true_obs_comp)
            cors_clr_comp = checkfitting.corr(Y_pred = Y_pred_clr,
                                              Y_obs = Y_obs_clr)
            Rsq_clr_comp = checkfitting.rsquared_feature(Y_pred = Y_pred_clr,
                                                         Y_obs = Y_obs_clr)
            mse_clr_comp = checkfitting.mse(Y_pred = Y_pred_clr,
                                       Y_obs = Y_obs_clr,
                                       sample_weight=w_test)   
            

            # Store per-bin results
            metrics_by_bin[fold_test] = {
                "cv": cv,
                "keep_mask": train_test.keep_mask,
                "selected_rank": int(cv.best_rank_),
                "selected_lambda": float(cv.best_lambda_),
                "mse_comp": mse_comp,
                "cors_comp": cors_comp,
                "Rsq_comp": Rsq_comp,
                "mse_clr_comp": mse_clr_comp,
                "cors_clr_comp": cors_clr_comp,
                "Rsq_clr_comp": Rsq_clr_comp,
                "Y_test_comp_df": checkfitting.convert_df(Y_test_comp_hat),
                "Y_obs_comp_df": checkfitting.convert_df(Y_true_obs_comp),
                "X_train_s": X_train_s,
                "X_test_s": X_test_s}
            
            # Optional: print summary
            if verbose == True:
                print(f"Bin {fold_test}: rank={cv.best_rank_}, lambda={cv.best_lambda_:.5g}, "
                    f"MSE clr comp={mse_clr_comp:.4f}")

        return metrics_by_bin