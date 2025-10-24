# src/myrrr/preprocessing.py
import numpy as np

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

class TrainTestSplit:
    def __init__(self, nfolds=5, random_state=0):
        self.nfolds = nfolds
        self.random_state = random_state

    def split(self, X, Y, fold):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        bins = rng.integers(1, self.nfolds + 1, size=n_samples)
        pred_mask = (bins == fold)
        train_mask = indices[~pred_mask]
        return (X[train_mask, :], X[pred_mask, :], Y[train_mask, :], Y[pred_mask, :], bins)
    
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
        Y = np.asarray(Y, dtype=float)
        return Y - self.mean_

    def inverse_transform(self, Yc):
        return Yc + self.mean_