# src/myrrr/preprocessing.py
from __future__ import annotations

import numpy as np

# controls star imports of the module (i.e., from regression.preprocessing import *)
__all__ = ["CompositionalILR", "StandardScalerX", "CenterY", "TrainTestSplit"]

try:
    from skbio.stats.composition import closure, multiplicative_replacement, ilr, ilr_inv
    SKBIO_AVAILABLE = True
except Exception:
    SKBIO_AVAILABLE = False

class CompositionalILR:
    def __init__(self, zero_replacement=True):
        self.zero_replacement = zero_replacement
        self.n_parts_ = None

    def fit(self, Y_comp):
        Y_comp = np.asarray(Y_comp, dtype=float)
        self.n_parts_ = Y_comp.shape[1]
        return self

    def transform(self, Y_comp):
        if not SKBIO_AVAILABLE:
            raise ImportError("skbio is required for CompositionalILR.")
        Y_comp = np.asarray(Y_comp, dtype=float)
        Y_closed = closure(Y_comp)
        Y_imp = multiplicative_replacement(Y_closed) if self.zero_replacement else Y_closed
        return ilr(Y_imp).astype(np.float64)

    def inverse_transform(self, Y_ilr):
        if not SKBIO_AVAILABLE:
            raise ImportError("skbio is required for CompositionalILR.")
        Y_ilr = np.asarray(Y_ilr, dtype=float)
        return ilr_inv(Y_ilr).astype(np.float64)

class CompositionalCLR:
    def __init__(self, zero_replacement=True):
        self.zero_replacement = zero_replacement
        self.n_parts_ = None    
    def fit(self, Y_comp):
        Y_comp = np.asarray(Y_comp, dtype=float)
        self.n_parts_ = Y_comp.shape[1]
        return self 
    def transform(self, Y_comp):
        if not SKBIO_AVAILABLE:
            raise ImportError("skbio is required for CompositionalCLR.")
        from skbio.stats.composition import clr, closure, multiplicative_replacement
        Y_comp = np.asarray(Y_comp, dtype=float)
        Y_closed = closure(Y_comp)
        Y_imp = multiplicative_replacement(Y_closed) if self.zero_replacement else Y_closed
        return clr(Y_imp).astype(np.float64)
    def inverse_transform(self, Y_clr):
        if not SKBIO_AVAILABLE:
            raise ImportError("skbio is required for CompositionalCLR.")
        from skbio.stats.composition import clr_inv
        Y_clr = np.asarray(Y_clr, dtype=float)
        return clr_inv(Y_clr).astype(np.float64)    

class TrainTestSplit:
    def __init__(self, nfolds=5, random_state=0):
        self.nfolds = nfolds
        self.random_state = random_state
        self.bins = None
        self.keep_mask = None

    def split(self, X, Y, fold):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else: 
            rng = np.random.default_rng()
        #generates random integers from 1 to nfolds for each sample, which determines the fold assignment for each sample
        self.bins = rng.integers(1, self.nfolds + 1, size=n_samples)
        pred_mask = (self.bins == fold)
        train_mask = indices[~pred_mask]
        return (X[train_mask, :], X[pred_mask, :], Y[train_mask, :], Y[pred_mask, :])
    
    def filter_feature(self, X_train, X_pred):
        finite_mask = np.isfinite(X_train).all(axis=0)
        nzv_mask = X_train.std(axis=0) > 0.0
        self.keep_mask = finite_mask & nzv_mask
        return X_train[:, self.keep_mask], X_pred[:, self.keep_mask]


class ImprovedTrainTestSplit:
    """
    Robust K-fold splitter with optional stratification, grouping, precompute, and validation split.

    Usage:
      - Stateless: construct with nfolds/random_state, call split(X, y=..., fold=...)
      - Precompute: set precompute=True and provide n_samples or y_for_precompute to compute splits at init.
    """
    def __init__(self,
                 nfolds: int = 5,
                 random_state: int | None = 42,
                 shuffle: bool = True,
                 precompute: bool = False,
                 n_samples: int | None = None,
                 y_for_precompute: np.ndarray | None = None,
                 ):
        if nfolds < 2:
            raise ValueError("nfolds must be >= 2")
        self.nfolds = int(nfolds)
        self.random_state = random_state
        self.shuffle = bool(shuffle)
        self._precomputed_splits = None
        # compatibility with original TrainTestSplit
        # `bins` stores fold assignment per sample: -1 means unassigned; 0..nfolds-1 are fold indices
        self.bins: np.ndarray | None = None
        # `keep_mask` stores boolean mask of kept features after filter_feature
        self.keep_mask: np.ndarray | None = None
        self.bins = None
        self.keep_mask = None


        if precompute:
            if y_for_precompute is not None:
                self._precomputed_splits = self._compute_splits_from_labels(y_for_precompute)
            elif n_samples is not None:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.random_state)
                self._precomputed_splits = list(kf.split(np.arange(n_samples)))
            else:
                raise ValueError("To precompute, provide `n_samples` or `y_for_precompute`.")

    def _compute_splits_from_labels(self, y: np.ndarray, groups: np.ndarray | None = None):
        y = np.asarray(y)
        n = len(y)
        if groups is not None:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=self.nfolds)
            return list(gkf.split(np.arange(n), groups=groups))
        # Prefer stratified when there are enough classes
        unique_classes = np.unique(y)
        if (y.ndim == 1) and (unique_classes.size >= min(self.nfolds, 2)):
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.random_state)
            return list(skf.split(np.arange(n), y))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.random_state)
            return list(kf.split(np.arange(n)))

    def split(self,
              X,
              y: np.ndarray | None = None,
              fold: int = 1,
              groups: np.ndarray | None = None,
              stratify: bool = False,
              return_indices: bool = False,
              val_split: float = 0.0,
              val_random_state: int | None = None,
              ):
        """
        Returns either (X_train, X_test, y_train, y_test) or (train_idx, test_idx) if `return_indices=True`.
        If `val_split>0`, returns (X_train, X_val, X_test, y_train, y_val, y_test).

        Note: `fold` is 1-based (1..nfolds). `self.bins` stores fold assignment using 1..nfolds.
        """
        from scipy import sparse
        from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedShuffleSplit

        n_samples = X.shape[0]
        if not (1 <= fold <= self.nfolds):
            raise ValueError(f"fold must be in [1, {self.nfolds}]")

        # obtain splits
        if self._precomputed_splits is not None:
            train_idx, test_idx = self._precomputed_splits[fold - 1]
            if len(train_idx) + len(test_idx) != n_samples:
                raise RuntimeError("Precomputed splits do not match number of samples in X")
        else:
            if groups is not None:
                splitter = GroupKFold(n_splits=self.nfolds)
                splits = list(splitter.split(np.arange(n_samples), groups=groups))
            elif stratify and (y is not None):
                splitter = StratifiedKFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.random_state)
                splits = list(splitter.split(np.arange(n_samples), y))
            else:
                splitter = KFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.random_state)
                splits = list(splitter.split(np.arange(n_samples)))
            train_idx, test_idx = splits[fold - 1]

        # build bins (fold assignment for every sample) for compatibility with prior API
        bins = np.full(n_samples, -1, dtype=int)
        if self._precomputed_splits is not None:
            for i, (_, te) in enumerate(self._precomputed_splits):
                # store 1-based fold labels
                bins[np.asarray(te, dtype=int)] = i + 1
        else:
            # `splits` exists in this branch
            for i, (_, te) in enumerate(splits):
                bins[np.asarray(te, dtype=int)] = i + 1
        self.bins = bins

        # validation split from train_idx
        val_idx = None
        if val_split > 0.0:
            train_idx_arr = np.asarray(train_idx)
            if (y is not None) and (np.asarray(y).ndim == 1):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=val_random_state or self.random_state)
                y_train = np.asarray(y)[train_idx_arr]
                sub_train_local, val_local = next(sss.split(train_idx_arr, y_train))
                final_train_idx = train_idx_arr[sub_train_local]
                val_idx = train_idx_arr[val_local]
            else:
                rng = np.random.default_rng(val_random_state or self.random_state)
                perm = rng.permutation(len(train_idx_arr))
                n_val = int(np.ceil(len(train_idx_arr) * val_split))
                val_idx = train_idx_arr[perm[:n_val]]
                final_train_idx = train_idx_arr[perm[n_val:]]
            train_idx = np.asarray(final_train_idx)
        else:
            train_idx = np.asarray(train_idx)

        if return_indices:
            return (train_idx, val_idx, np.asarray(test_idx)) if val_idx is not None else (train_idx, np.asarray(test_idx))

        def _slice(arr, idx):
            if sparse.issparse(arr):
                return arr[idx, :]
            return arr[idx]

        X_train = _slice(X, train_idx)
        X_test = _slice(X, test_idx)
        y_train = None if y is None else np.asarray(y)[train_idx]
        y_test = None if y is None else np.asarray(y)[test_idx]

        if val_idx is not None:
            X_val = _slice(X, val_idx)
            y_val = None if y is None else np.asarray(y)[val_idx]
            return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_test, y_train, y_test

    def filter_feature(self, X_train, X_pred, var_threshold: float = 0.0):
        """
        Sparse-aware feature filter. Computes finite & non-zero-variance mask across columns.
        Sets `self.keep_mask` and returns (X_train_filtered, X_pred_filtered).

        Parameters
        - X_train: array-like or scipy.sparse matrix of shape (n_samples_train, n_features)
        - X_pred: array-like or scipy.sparse matrix of shape (n_samples_pred, n_features)
        - var_threshold: minimum variance threshold to keep a feature
        """
        from scipy import sparse

        if sparse.issparse(X_train):
            # compute column mean and variance in a sparse-friendly way
            mean = np.asarray(X_train.mean(axis=0)).ravel()
            X2 = X_train.copy()
            X2.data **= 2
            mean_sq = np.asarray(X2.mean(axis=0)).ravel()
            var = mean_sq - mean ** 2
            finite_mask = np.isfinite(mean) & np.isfinite(var)
            nzv_mask = var > 0.0
            keep_mask = finite_mask & nzv_mask & (var > var_threshold)
        else:
            X_train = np.asarray(X_train, dtype=float)
            finite_mask = np.isfinite(X_train).all(axis=0)
            nzv_mask = X_train.std(axis=0) > 0.0
            keep_mask = finite_mask & nzv_mask & (X_train.var(axis=0) > var_threshold)

        self.keep_mask = np.asarray(keep_mask, dtype=bool)

        # apply mask (sparse-aware slicing)
        if sparse.issparse(X_train):
            return X_train[:, self.keep_mask], X_pred[:, self.keep_mask]
        else:
            return X_train[:, self.keep_mask], X_pred[:, self.keep_mask]


class StandardScalerX:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Xs):
        return Xs * self.std_ + self.mean_

class CenterY:
    def __init__(self):
        self.mean_ = None

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        self.mean_ = Y.mean(axis=0, keepdims=True)
        return self

    def transform(self, Y):
        return Y - self.mean_
        
    def fit_transform(self, Y):
        return self.fit(Y).transform(Y)

    def inverse_transform(self, Yc):
        return Yc + self.mean_

import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse import spmatrix, sparray
from scipy.sparse import hstack, csr_matrix, spmatrix
from sklearn.preprocessing import normalize
from typing import Literal, Dict, List

class build_pseudobulk_matrix:
    def __init__(self, X_list: list[spmatrix|sparray], folds: Dict[str, List[int]]):
        '''
        Docstring for __init__
        
        :param X_list: a list of sparse gene expression matrices (cells x genes)
        :param folds: a dictionary storing train/val/test splits
        '''
        self.X_list = X_list
        self.folds = folds

    def construct(self, fold: int, method: Literal["sum_log", "log_mean"] = "sum_log",split: Literal["train", "val", "test"] = "train", scale_factor: float = 1e4):
        ''' 
        Construct pseudobulk matrix (sample by genes) for a given fold and split
        method: "sum_log" or "log_mean". "sum_log": sum the raw counts across cells in each sample, then log-normalize the pseudobulk counts;
                "log_mean": log-normalize the counts per cell first, then take the mean across cells in each sample
        fold: which fold to select from self.folds
        split: which split to select from the fold (train/val/test)
        Return: a pseudobulk sparse matrix (samples x genes) 
        '''
        if method not in ["sum_log", "log_mean"]:
            raise ValueError(f"Method {method} not recognized. Choose from 'sum_log' or 'log_mean'.")
        if method == "sum_log":
            # fold[split] stores a list of sample indices in X_list for pseudobulk aggregation
            # X = [self.X_list[i] for i in self.folds[fold][split]]
            # if sp.sparse.issparse(X[0]):    
            #     X = sp.sparse.vstack(X)
            # else:
            #     X = [sp.sparse.csr_matrix(x) for x in X]
            #     X = sp.sparse.vstack(X)

            # select sample indices for this fold/split
            idxs = self.folds[fold][split]
            
            # quick empty check
            if len(idxs) == 0:
                # return an empty (nsamples x ngenes) CSR of appropriate shape
                # we cannot infer nsamples here, caller will handle an empty case; return empty csr (0 x ngenes)
                return sp.sparse.csr_matrix((0, self.X_list[0].shape[1]), dtype=np.float32)
            
            # pre-convert all per-sample matrices to CSR float32 once and cache on object
            if not hasattr(self, "_X_csr"):
                _csr_list = []
                for x in self.X_list:
                    if sp.sparse.issparse(x):
                        # tocsr() is cheap if already CSR; astype(copy=False) avoids copy if already float32
                        _csr_list.append(x.tocsr().astype(np.float32, copy=False))
                    else:
                        _csr_list.append(sp.sparse.csr_matrix(x, dtype=np.float32))
                self._X_csr = _csr_list
            
            # gather references for requested indices (no copies)
            sel = [self._X_csr[i] for i in idxs]
            
            # helper: chunked vstack to limit peak memory for very many small matrices
            def _chunked_vstack(mats, chunk_size: int = 100):
                if len(mats) <= chunk_size:
                    return sp.sparse.vstack(mats, format="csr")
                parts = []
                cur = []
                for m in mats:
                    cur.append(m)
                    if len(cur) >= chunk_size:
                        parts.append(sp.sparse.vstack(cur, format="csr"))
                        cur = []
                if cur:
                    parts.append(sp.sparse.vstack(cur, format="csr"))
                if len(parts) == 1:
                    return parts[0]
                return sp.sparse.vstack(parts, format="csr")
            
            # stack once (adjust chunk_size to match memory / number of CPUs)
            X = _chunked_vstack(sel, chunk_size=128)
            
            codes = self.build_cell_ids_samples(self.X_list, fold=self.folds[fold], split=split)
            G = self.build_sample_by_cell_matrix(nsamples = len(self.folds[fold][split]),
                                                ncells = X.shape[0],
                                                codes = codes)
            Xs_pseudobulk = self.do_pseudobulk(G, X)
            Xs_pseudobulk = self.log_normalize_sparse(Xs_pseudobulk, scale_factor=scale_factor)

        else:  # method == "log_mean"
            X_log_normed = self.log_normalize(fold=fold, split=split, scale_factor=scale_factor)
            codes = self.build_cell_ids_samples(self.X_list, fold=self.folds[fold])
            G = self.build_sample_by_cell_matrix(nsamples = len(self.folds[fold][split]),
                                                ncells = X_log_normed.shape[0],
                                                codes = codes)
            # Number of cells per sample/group
            # reshape to 2D matrix
            n_cells_per_group = np.bincount(codes).reshape(-1, 1)
            Xs_pseudobulk = self.do_pseudobulk(G, X_log_normed)
            Xs_pseudobulk = Xs_pseudobulk.multiply(1 / n_cells_per_group)
        
        return Xs_pseudobulk
    
    def log_normalize(self, fold, split: Literal["train", "val", "test"] = "train", scale_factor: float = 1e4):
        '''
        Log-normalize a vertically concatenated sparse matrix from a list of sparse matrices
        fold: which fold to select from self.folds
        split: which split to select from the fold (train/val/test)
        Return: a log-normalized sparse matrix  
        '''
        # select a fold
        fold = self.folds[fold]
        # fold[split] stores a list of sample indices in X_list for pseudobulk aggregation
        X = [self.X_list[i] for i in fold[split]]
        if sp.sparse.issparse(X[0]):    
            X = sp.sparse.vstack(X)
        else:
            X = [sp.sparse.csr_matrix(x) for x in X]
            X = sp.sparse.vstack(X)
        
        X = self.log_normalize_sparse(X, scale_factor=scale_factor)
        return X
    
    @staticmethod
    def log_normalize_sparse(X: spmatrix|sparray, scale_factor: float = 1e4):
        '''
        Log-normalize a sparse matrix
        Return: a log-normalized sparse matrix
        ''' 
        if not sp.sparse.issparse(X):
            print(f"Input matrix is not sparse. Converting to sparse matrix.")
            X = sp.sparse.csr_matrix(X)
        
        # Normalize rows to sum to one
        X_norm = normalize(X, norm = 'l1', axis = 1)
        # Log transform non-zero values only
        X_log = X_norm.copy()
        X_log.data = np.log1p(X_log.data * scale_factor)
        return X_log
    
    @staticmethod
    def build_cell_ids_samples(Xs_raw, fold, split: Literal["train", "val", "test"] = "train"):
        '''
        Build list of sample IDs for each cell

        Return: 
        codes: a list storing sample IDs (also columns) each cells belong to
        (e.g., [0,0,0,1,1] means first column has 3 cells in sample 0, second column has 2 cells in sample 1
        '''
        ncells_list = [Xs_raw[i].shape[0] for i in fold[split]]
        # codes = []
        # for j, ncell in enumerate(ncells_list):
        #     codes.extend([j] * ncell)
        codes = np.repeat(np.arange(len(ncells_list), dtype = int), ncells_list)
        return codes
    
    @staticmethod
    def build_sample_by_cell_matrix(nsamples, ncells, codes):
        '''
        nsamples: number of samples
        ncells: number of cells
        codes: id of each cells belonging to a particular sample

        Return:
        G: a sparse matrix where rows are cells, columns are sample IDs
        (e.g., array[[1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0])
        '''
        G = sp.sparse.csr_matrix(
                (np.ones(ncells, dtype=np.float32), (np.arange(ncells), codes)),
                shape=(ncells, nsamples)
            )
        return G
    
    @staticmethod
    def do_pseudobulk(G, X):
        '''
        G: sample by cell matrix indicating which cells belong to which sample
        X: sparse gene expression matrix per sample
        '''
        return G.T@X



    @staticmethod
    def build_X_design(X: spmatrix, 
                        Z: pd.DataFrame, 
                        fold: Dict[str, List[int]], 
                        split: Literal["train", "val", "test"] = "train") -> spmatrix:
        '''
        X: pseudobulk, sparse sample by gene matrix
        Z: covariate matrix (e.g., batch information, age)

        Return:
        X_design: concatenate X and Z together to get a sparse matrix
        '''
        Z_sparse = csr_matrix(Z.to_numpy()[fold[split], :])
        X_design = hstack([X, Z_sparse])
        return X_design

    @staticmethod
    def X_var_matrix(X):
        X_mean = X.toarray().mean(axis = 0)
        X_centered = X.toarray() - X_mean
        return X_centered ** 2


# do batch correction

import numpy as np
import pandas as pd

def make_design_matrix(meta_df, covariates=("sex", "batch"), add_intercept=True):
    Z = pd.get_dummies(meta_df.loc[:, covariates], drop_first=True)
    if add_intercept:
        Z.insert(0, "Intercept", 1.0)
    return Z

def residualize_matrix(X_df, Z_df, keep_mean=True):
    """
    X_df: samples x genes (logCPM)
    Z_df: samples x covariates (with intercept column recommended)

    Returns:
      X_corr_df: samples x genes with covariate effects removed
    """
    # Align samples
    common = X_df.index.intersection(Z_df.index)
    X = X_df.loc[common].to_numpy(dtype=float)
    Z = Z_df.loc[common].to_numpy(dtype=float)

    # Fit B = (Z^T Z)^-1 Z^T X  via lstsq (stable)
    B, *_ = np.linalg.lstsq(Z, X, rcond=None)     # shape: (n_cov x n_genes)

    # Residuals
    X_hat = Z @ B
    R = X - X_hat

    if keep_mean:
        # add back gene means to keep on a comparable scale
        R = R + X.mean(axis=0, keepdims=True)

    return pd.DataFrame(R, index=common, columns=X_df.columns)


# combine everything into a single class that can run the whole pipeline, with options for folds and covariate correction

import argparse
import os
import re
from typing import Literal, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import spmatrix, sparray, hstack, csr_matrix
from sklearn.preprocessing import normalize

try:
    import scanpy as sc  # type: ignore
except Exception:  # pragma: no cover
    sc = None

try:
    import rapids_singlecell as rsc  # type: ignore
except Exception:  # pragma: no cover
    rsc = None


# --- Helpers (unchanged) ---
def normalize_cell_types(arg) -> List[str]:
    if isinstance(arg, str):
        parts = [s.strip(" '\"") for s in re.split(r"[,\\s]+", arg) if s.strip()]
        return parts if parts else []
    elif isinstance(arg, (list, tuple)):
        return [str(s).strip(" '\"") for s in arg]
    else:
        raise TypeError(f"Unsupported type for cell_type: {type(arg)}")


def filter_method_max_counts(adata, max_counts: int = 3):  # unchanged
    max_counts_per_gene = np.max(adata.X, axis=0).toarray().flatten()
    gene_mask = max_counts_per_gene >= max_counts
    print(f"Number of genes to keep = {gene_mask.sum()}")
    return adata[:, gene_mask].copy()


def filter_method_min_cells(adata, min_cells: int = 3):  # unchanged
    if rsc is None:
        raise ImportError("rapids_singlecell is required for method='min_cells'.")
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.filter_genes(adata, min_cells=min_cells)
    rsc.get.anndata_to_CPU(adata)
    return adata.copy()


# --- Unified Class ---
class PseudobulkProcessor:
    """
    Full pipeline: subset AnnData by cell type → pseudobulk → (optional) folds/splits → (optional) batch correction.

    Examples:
        # Simple mode (no folds): log-CPM DataFrame per sample
        proc = PseudobulkProcessor(adata_path="...", cell_type="HSC_MPP", out_dir="...")
        log_cpm_df = proc.build_and_correct(correct_covariates=False)

        # Folded mode: sparse pseudobulk per fold/split
        folds = {"fold0": {"train": [0,1], "val": [2], "test": [3]}}
        proc.folds = folds
        sparse_pb = proc.build_and_correct(method="sum_log", split="train")
    """

    def __init__(
        self,
        adata: sc.AnnData,
        covariate_csv: str = None,
        out_dir: str = "output",
        method: Literal["max_counts", "min_cells"] = "max_counts",
        cell_type: str = "HSC_MPP",
        min_cells: int = 3,
        lower_pct: float = 5.0,
        upper_pct: float = 95.0,
        folds: Optional[Dict[str, Dict[str, List[int]]]] = None,  # e.g. {"fold0": {"train": [0,1], ...}}
        pb_method: Literal["sum_log", "log_mean"] = "sum_log",
        scale_factor: float = 1e6,  # CPM=1e6, scTransform-like=1e4
        correct_covariates: Optional[List[str]] = None,  # e.g. ["sex_src", "ages"]
        cell_type_list: List[str] = [],
        sample_column: str = "sample_id",
    ):
        self.adata = adata
        self.covariate_csv = covariate_csv
        self.out_dir = Path(out_dir)
        self.method = method
        self.cell_type_arg = cell_type
        self.min_cells = min_cells
        self.lower_pct, self.upper_pct = lower_pct, upper_pct
        self.folds = folds
        self.pb_method = pb_method
        self.scale_factor = scale_factor
        self.correct_covariates = correct_covariates 
        self.cell_type_list = cell_type_list
        self.sample_column = sample_column
        # Internal state
        self.selected_cell_type_list: List[str] = []
        self.cells_by_samples: Dict[str, List[str]] = {}
        self.cell_type_proportions: pd.DataFrame = pd.DataFrame()
        self.meta: pd.DataFrame = pd.DataFrame()
        self.log_cpm_df: pd.DataFrame = pd.DataFrame()
        self.X_list: List[spmatrix] = []  # For folded mode

    def build_pipeline(self, feature_filter_conditions: dict, sample_filter_list, subset_list, groupby_list, save_proportions: bool = True) -> Union[pd.DataFrame, spmatrix]:
        """Main entry: filter → subset → pseudobulk → return DataFrame/sparse."""
        self._load_and_filter()
        # for filter HSC_MPP in samples of interest
        self._subset_and_prepare(feature_filter_conditions, subset_list)
        # for filter samples of interest      
        if sample_filter_list is not None:
            missing = set(sample_filter_list) - feature_filter_conditions.keys()
            if missing:
                raise KeyError(f"Unknown keys not in feature_filter_conditions: {sorted(missing)}")

            sample_filter_conditions = {k: self.feature_filter_conditions[k] for k in sample_filter_list}
            # combine all filter conditions into one mask
            filter_condition = np.logical_and.reduce(list(sample_filter_conditions.values()))
            self._compute_proportions_and_meta(groupby_list, filter_condition)
        else:
            self._compute_proportions_and_meta(groupby_list)

        if self.folds is None:
            # Use all samples together. 
            self.X_list = self._build_xlist_from_samples()
            n_samples = len(self.X_list) #or len(self.cells_by_samples) 
            self.folds =  {'fold0': {'all': np.arange(n_samples)}}
            # result = build_pseudobulk_matrix(self.X_list, self.folds).log_normalize_sparse(self.adata.X, scale_factor=self.scale_factor)
            result = self._build_sparse_pseudobulk(split = "all")

            # result = self._apply_correction(self.log_cpm_df)
            # self.log_cpm_df = result
        else:
            # Folded: sparse pseudobulk
            self.X_list = self._build_xlist_from_samples()
            result = self._build_sparse_pseudobulk()
            # result = self._apply_correction(result)  # Sparse version if needed

        self._save_results(result)
        return result

    # --- Pipeline Steps (public for debugging) ---
    def _load_and_filter(self) -> None:
        """Load + gene/cell filtering."""
        # self.adata = sc.read(self.adata_path)
        if self.method == "max_counts":
            self.adata = filter_method_max_counts(self.adata)
        else:
            self.adata = filter_method_min_cells(self.adata, self.min_cells)

        gene_counts_per_cell = self.adata.X.sum(axis=1).A1
        q_low, q_high = np.percentile(gene_counts_per_cell, [self.lower_pct, self.upper_pct])
        print(f"Counts: {self.lower_pct}%={q_low}, {self.upper_pct}%={q_high}")
        mask = (gene_counts_per_cell > q_low) & (gene_counts_per_cell < q_high)
        self.adata = self.adata[mask].copy()

    def _subset_and_prepare(self, feature_filter_conditions: dict, groupby_list) -> None:
        """Covariates + sample_id + counts layer."""

        if self.covariate_csv is not None:
            # for the case of HSC_MPP in normal samples
            df_cov = pd.read_csv(self.covariate_csv)
            self.adata.obs = (
                self.adata.obs.reset_index() # index (cell barcodes) becomes a column called 'index'
                .merge(df_cov, on="indiv_id", how="left") #merge
                .set_index("index") #restore original row name as index
            )
            self.adata.obs.index.name = None

            self.adata.obs["sex_src"] = self.adata.obs["sexes"].astype(str) + "_" + self.adata.obs["src"].astype(str)
            self.adata.obs["sex_src"].astype("category")
            self.adata.layers["counts"] = self.adata.X.copy()

            self.adata.obs["sample_id"] = (
                self.adata.obs["exp_name"].astype("string").str.cat(self.adata.obs["indiv_id"].astype("string"), sep="_")
            )
            self.adata.obs["sample_id"].astype("category")

            # Cell type subset (HSC_MPP special case) for generating gene expression matrix
            if self.cell_type_arg == "HSC_MPP":
                self.selected_cell_type_list = self.cell_type_arg.split("_")
            else:
                self.selected_cell_type_list = normalize_cell_types(self.cell_type_arg)
            mask = self.adata.obs["cell_type"].isin(self.selected_cell_type_list)
            ad_obs = self.adata.obs.loc[mask]
            self.cells_by_samples = ad_obs.groupby("sample_id").apply(lambda g: sorted(g.index)).to_dict()
        else:
            self.feature_filter_conditions = {k: self.adata.obs[k].isin(v) for k, v in feature_filter_conditions.items()}
            # combine all filter conditions into one mask
            condition_mask = np.logical_and.reduce(list(self.feature_filter_conditions.values()))

            cells_by_samples = (self.adata.obs.loc[condition_mask]
                                .groupby(groupby_list)
                                .apply(lambda g: sorted(g.index))
                                .to_dict())
            self.cells_by_samples = cells_by_samples

    def _compute_proportions_and_meta(self, groupby_list, filter_condition = None) -> None:
        """Cell proportions + metadata."""
        if filter_condition is None:
            ct_counts = self.adata.obs.groupby(groupby_list).size().unstack(fill_value=0)
            self.meta = self.adata.obs.drop_duplicates("sample_id").set_index("sample_id")[["sex_src", "ages"]]
            self.meta["sex_src"] = self.meta["sex_src"].astype("category")
            self.meta["ages"] = self.meta["ages"].astype(int)

        else:
            ct_counts = self.adata.obs.loc[filter_condition].groupby(groupby_list).size().unstack(fill_value=0)
            self.meta = self.adata.obs.loc[filter_condition].drop_duplicates(self.sample_column).set_index(self.sample_column)[self.correct_covariates]

        if self.cell_type_list is not None:
            ct_counts = ct_counts.loc[:, self.cell_type_list]
        
        self.cell_type_proportions = ct_counts.div(ct_counts.sum(axis=1), axis=0)
        self.meta = self.meta.reindex(self.cell_type_proportions.index)

        


    def _build_xlist_from_samples(self) -> List[spmatrix]:
        """Convert samples → X_list for folds."""
        assert [dict_key[0] if isinstance(dict_key, tuple) else dict_key for dict_key in list(self.cells_by_samples.keys())] == self.cell_type_proportions.index.tolist(), "Sample keys in cells_by_samples should match sample order in cell_type_proportions"

        from scipy import sparse

        # Pull the full counts matrix once; avoid per-sample AnnData slicing.
        counts = self.adata.layers.get("counts", self.adata.X)
        counts_is_sparse = sparse.issparse(counts)
        if counts_is_sparse and not sparse.isspmatrix_csr(counts):
            counts = counts.tocsr()

        # Prefer AnnData's obs_names when available; otherwise fall back to obs.index.
        cell_index = getattr(self.adata, "obs_names", None)
        if cell_index is None:
            cell_index = self.adata.obs.index

        pids = list(self.cells_by_samples.keys())

        # Batch index lookup: one get_indexer call instead of one per sample.
        lengths: List[int] = []
        flat_ids: List[str] = []
        for pid in pids:
            ids = self.cells_by_samples[pid]
            lengths.append(len(ids))
            flat_ids.extend(ids)

        row_pos_all = cell_index.get_indexer(flat_ids)
        if row_pos_all.size and (row_pos_all < 0).any():
            bad = np.flatnonzero(row_pos_all < 0)
            examples = [flat_ids[i] for i in bad[:5]]
            raise KeyError(f"{bad.size} cell ids not found in adata.obs_names; examples: {examples}")

        X_list: List[spmatrix] = []
        offset = 0
        for n in lengths:
            row_pos = row_pos_all[offset : offset + n]
            offset += n
            if counts_is_sparse:
                X_list.append(counts[row_pos, :])
            else:
                # Dense fallback: convert per-sample slice to CSR.
                X_list.append(sparse.csr_matrix(counts[row_pos, :]))
        return X_list

    def _build_sparse_pseudobulk(self, fold: str = "fold0", split: str = "train") -> spmatrix:
        """Folded sparse pseudobulk (from your class)."""
        pb = build_pseudobulk_matrix(self.X_list, self.folds)  # Reuse logic
        return pb.construct(fold=fold, method=self.pb_method, split=split, scale_factor=self.scale_factor)

    def _apply_correction(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Dense correction."""
        if not self.correct_covariates:
            return X_df
        Z_df = make_design_matrix(self.meta, self.correct_covariates)
        return residualize_matrix(X_df, Z_df)

    def _save_results(self, result) -> None:
        self.out_dir.mkdir(exist_ok=True)
        self.cell_type_proportions.to_csv(self.out_dir / f"proportions_{self.method}.csv")
        self.meta.to_csv(self.out_dir / f"metadata_{self.method}.csv")
        if isinstance(result, pd.DataFrame):
            result.to_csv(self.out_dir / f"log_cpm_{self.cell_type_arg}_{self.method}.csv")


# --- CLI (extended) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (your args + --folds_json --correct_covariates "sex_src,ages")
    args = parser.parse_args()
    proc = PseudobulkProcessor(
        adata_path=args.adata_path,
        covariate_csv=args.covariate_csv,
        out_dir=args.path_dir,
        cell_type=args.cell_type,
        method=args.method,
        correct_covariates=args.correct_covariates.split(",") if args.correct_covariates else None,
    )
    result = proc.build_and_correct()
