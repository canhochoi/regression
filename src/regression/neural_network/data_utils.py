import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

# A simple bulk head to map pseudobulk (S, G) to proportions (S, T)
class PseudobulkAggregator(nn.Module):
    def __init__(self, mode = "sum"):
        super().__init__()
        assert mode in ("sum", "mean")
        self.mode = mode

    def forward(self, X_batch: torch.Tensor, batch_idx: torch.LongTensor) -> torch.Tensor:
        '''
        X_batch: stacked matrices of cells by genes 
        batch_idx: list of batch ids
        '''
        # number of samples
        S = int(batch_idx.max().item()) + 1 # index starts from 0, 1, 2
        # number of genes
        G = X_batch.shape[1] 
        X_bulk = torch.zeros((S,G), device = X_batch.device, dtype=X_batch.dtype)
        # sum gene counts for all cells in a sample
        X_bulk.index_add_(0, batch_idx, X_batch)
        if self.mode == "mean":
            counts = torch.bincount(batch_idx, minlength = S).clamp_min(1).unsqueeze(1) 
            X_bulk = X_bulk / counts
        return X_bulk

class RowLibSizeNorm(nn.Module):
    """
    Row-wise library-size normalization: (S, G) -> (S, G)
    y_s = (x_s / (sum_j x_{s,j} + eps)) * scale
    """
    def __init__(self, scale: float = 1e6, eps: float = 1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, X_bulk: torch.Tensor) -> torch.Tensor:
        lib = X_bulk.sum(dim=1, keepdim=True) + self.eps
        return torch.log1p((X_bulk / lib) * self.scale)

# ----------------------------------------
# for minibatch of single cell gene expression data
# ----------------------------------------


from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SampleCellsDataset(Dataset):
    """
    Sample-level dataset: one item = all cells from one sample.

    Xs_raw: list of scipy.sparse CSR matrices or dense arrays (n_cells_i, G)
    Z: np.ndarray or torch.Tensor of shape (S, C) — sample-level covariates
    Y: np.ndarray or torch.Tensor of shape (S, T) — sample-level targets (ILR or composition)

    __getitem__(idx) returns:
      Xi: (n_cells_i, G) torch.float32
      idx: int (dataset sample index)
      Zi: (C,) torch.float32
      Yi: (T,) torch.float32
    """
    def __init__(self, Xs_raw: List, Z, Y, dtype: torch.dtype = torch.float32):
        self.Xs_raw: List[torch.Tensor] = []

        # Convert each sample's sparse/dense matrix to a contiguous float32 torch.Tensor
        for Xi_raw in Xs_raw:
            # Handle scipy sparse
            if hasattr(Xi_raw, "toarray"):
                Xi_np = np.ascontiguousarray(Xi_raw.toarray().astype(np.float32, copy=False))
            else:
                Xi_np = np.ascontiguousarray(np.asarray(Xi_raw, dtype=np.float32))
            self.Xs_raw.append(torch.from_numpy(Xi_np).to(dtype=dtype))

        # Store Z, Y as torch tensors
        Z_np = np.asarray(Z, dtype=np.float32)
        Y_np = np.asarray(Y, dtype=np.float32)
        self.Z = torch.from_numpy(Z_np).to(dtype=dtype)
        self.Y = torch.from_numpy(Y_np).to(dtype=dtype)

        self.dtype = dtype
        assert len(self.Xs_raw) == len(self.Z) == len(self.Y), "Length mismatch among Xs_raw, Z, Y"

    def __len__(self) -> int:
        return len(self.Xs_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        Xi = self.Xs_raw[idx]     # (n_cells_i, G)
        Zi = self.Z[idx]          # (C,)
        Yi = self.Y[idx]          # (T,)
        return Xi, idx, Zi, Yi

# old and slow collate function for SampleCellsDataset
def collate_samples_full_Z_cell(
    batch: List[Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for SampleCellsDataset.

    Input batch: list of tuples (Xi, idx, Zi, Yi), length B samples.
    Returns:
      X_batch           : (sum_i n_cells_i, G), stacked cells across samples
      cell_to_batch     : (sum_i n_cells_i,), torch.long dataset sample indices per cell
      Z_batch           : (sum_i n_cells_i, C), per-cell replicated covariates
      Y_batch           : (B, T), sample-level targets in the batch order
      sample_idx_batch  : (B,), torch.long dataset sample indices in batch row order

    Notes:
    - cell_to_batch carries dataset indices; if your aggregator expects local 0..B-1,
      remap inside the aggregator (or change this to store local j IDs).
    """
    n_cells_list = [Xi.shape[0] for (Xi, _, _, _) in batch]
    total_cells = int(sum(n_cells_list))
    G = batch[0][0].shape[1]
    C = batch[0][2].numel()
    T = batch[0][3].numel()

    X_batch = torch.empty((total_cells, G), dtype=batch[0][0].dtype)
    cell_to_batch = torch.empty((total_cells,), dtype=torch.long)
    Z_batch = torch.empty((total_cells, C), dtype=batch[0][2].dtype)
    Y_batch = torch.empty((len(batch), T), dtype=batch[0][3].dtype)
    sample_idx_batch = torch.empty((len(batch),), dtype=torch.long)

    offset = 0
    for j, (Xi, idx, Zi, Yi) in enumerate(batch):
        n = Xi.shape[0]
        X_batch[offset:offset+n].copy_(Xi)
        cell_to_batch[offset:offset+n] = idx
        Z_batch[offset:offset+n].copy_(Zi.unsqueeze(0).repeat(n, 1))
        Y_batch[j].copy_(Yi)
        sample_idx_batch[j] = idx
        offset += n

    return X_batch, cell_to_batch, Z_batch, Y_batch, sample_idx_batch

# new and faster collate function for SampleCellsDataset
def collate_samples_compact(batch):
    '''
    Output:
    X_cells: (N_cells, G) stacked cell-gene matrices
    Z_sample: (B, C) sample-level covariates
    Y_batch: (B, T) sample-level targets
    sample_idx_batch: (B,) dataset sample indices in batch row order
    cell_to_batch: a list following 0, 1, 2, ... in accordance to ordering in sample_idx_batch
    '''
    # batch: list of tuples (Xi, idx, Zi, Yi)
    Xs, bidxs, Zs, Ys, sidxs = [], [], [], [], []
    for j, (Xi, idx, Zi, Yi) in enumerate(batch):
        n = Xi.shape[0]
        Xs.append(Xi)  # (n, G)
        bidxs.append(torch.full((n,), j, dtype=torch.long))  # local 0..B-1 IDs
        Zs.append(Zi.unsqueeze(0))  # (1, C), per-sample
        Ys.append(Yi.unsqueeze(0))  # (1, T), per-sample
        sidxs.append(torch.tensor(idx, dtype=torch.long).unsqueeze(0))  # dataset index
    X_cells = torch.cat(Xs, dim=0)               # (N_cells, G)
    cell_to_batch = torch.cat(bidxs, dim=0)      # (N_cells,)
    Z_sample = torch.cat(Zs, dim=0)              # (B, C)
    Y_batch = torch.cat(Ys, dim=0)               # (B, T)
    sample_idx_batch = torch.cat(sidxs, dim=0)   # (B,)
    return X_cells, cell_to_batch, Z_sample, Y_batch, sample_idx_batch

# ---------------------------------------------
# Split utilities and dataset builders
# ---------------------------------------------
def split_indices(
    S: int,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Random split of S samples into train/val/test indices.
    Returns dict with keys 'train', 'val', 'test'.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(S)
    n_test = int(round(S * test_frac))
    n_val = int(round(S * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_datasets(
    method: str,
    Xs_raw: List,
    Z: Optional[Union[pd.DataFrame, np.ndarray]],                        
    Y_imp_ilr: np.ndarray,
    cell_type_proportions_df: pd.DataFrame,  
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """
    Construct train/val/test SampleCellsDataset objects based on method.
    
    Xs_raw: list of S sample cell-gene matrices
    Z: sample-level covariates (S, C)
    Y_imp: ILR-transformed targets (S, T) or CLR-transformed or None
    cell_type_proportions_df: cell-type proportions dataframe (S, T) 

    method:
      - endswith('ilr'): targets are ILR (Y_imp).
      - endswith('clr'): targets are CLR (Y_imp).
      - else: targets are cell-type compositions (cell_type_proportions_df).
      - special 'ilr_recenter': subtract training mean from ILR and use centered targets.

    Returns:
      datasets: dict with 'train', 'val', 'test' keys and SampleCellsDataset values
      extras  : dict with optional 'Y_train_mean' (for ilr_recenter)
    """
    S = len(Xs_raw)
    idxs = split_indices(S, test_frac=test_frac, val_frac=val_frac, seed=seed)

    # Convert Z to numpy array if it is a pandas DataFrame
    Z_np = np.asarray(Z, dtype=np.float32)

    datasets = {}
    extras = {}

    if method == "ilr_recenter":
        assert Y_imp_ilr is not None, "Y_imp_ilr must be provided for ilr_recenter."
        Y_imp_ilr_np = np.asarray(Y_imp_ilr, dtype=np.float32)
        Y_train_mean = Y_imp_ilr_np[idxs["train"], :].mean(axis=0)
        Y_centered = Y_imp_ilr_np - Y_train_mean
        extras["Y_train_mean"] = Y_train_mean
        for split in ("train", "val", "test"):
            idx = idxs[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_centered[idx, :],
                dtype=dtype,
            )
        return datasets, extras, idxs

    if method.endswith("ilr"):
        assert Y_imp_ilr is not None, "Y_imp_ilr must be provided for ILR methods."
        Y_ilr_np = np.asarray(Y_imp_ilr, dtype=np.float32)
        for split in ("train", "val", "test"):
            idx = idxs[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_ilr_np[idx, :],
                dtype=dtype,
            )
        # Optionally also compute training mean and std if you plan to recenter in loss
        extras["Y_train_mean"] = Y_ilr_np[idxs["train"], :].mean(axis=0)
        extras["Y_train_std"] = Y_ilr_np[idxs["train"], :].std(axis=0)

    elif method.endswith("celltype"):
        # Cell-type proportions
        assert cell_type_proportions_df is not None, "cell_type_proportions_df must be provided for composition methods."
        Y_comp_np = np.asarray(cell_type_proportions_df, dtype=np.float32)
        for split in ("train", "val", "test"):
            idx = idxs[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_comp_np[idx, :],
                dtype=dtype,
            )
    elif method.endswith("clr"):
        # Cell-type proportions with CLR targets
        assert Y_imp_ilr is not None, "Y_imp_ilrmust be provided for composition methods."
        Y_clr_np = np.asarray(Y_imp_ilr, dtype=np.float32)
        for split in ("train", "val", "test"):
            idx = idxs[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_clr_np[idx, :],
                dtype=dtype,
            )

    return datasets, extras, idxs


def build_loaders(
    datasets: Dict[str, SampleCellsDataset],
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    collate_fn=collate_samples_compact,
    seed: int = 0,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for train/val/test splits from datasets.
    """    
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if num_workers >= 2:
        import random
        def seed_worker(worker_id):
            # Make worker-level RNG reproducible
            base_seed = torch.initial_seed() % 2**32
            np.random.seed(base_seed + worker_id)
            random.seed(base_seed + worker_id)
        train_data = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, generator=gen,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )
    else:
        train_data = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, generator=gen,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
        )
    return {
        "train": train_data,
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
        ),
    }

def compute_cellwise_stats(train_loader, log_norm: bool = True, scaling_factor = 1e4, device="cpu"):
    gene_sum = gene_sumsq = cov_sum = cov_sumsq = None
    n_cells_total = 0
    for X_batch, cell_to_batch, Z_batch, Y_batch, _ in train_loader:
        Xb = X_batch.to(device=device, dtype=torch.float32)
        Zb = Z_batch.to(device=device, dtype=torch.float32)
         # add this to map Z to each cell 
        Zb = Zb[cell_to_batch.to(device = device)]

        if log_norm:
            Xb = torch.log1p(Xb / Xb.sum(dim=1, keepdim=True) * scaling_factor)
        else:
            pass # no normalization
        
        if gene_sum is None:
            G = Xb.shape[1]
            # C = Zb.shape[1]
            gene_sum   = torch.zeros(G, dtype=torch.float32, device=device)
            gene_sumsq = torch.zeros(G, dtype=torch.float32, device=device)
            # cov_sum    = torch.zeros(C, dtype=torch.float32, device=device)
            # cov_sumsq  = torch.zeros(C, dtype=torch.float32, device=device)
        gene_sum   += Xb.sum(dim=0)
        gene_sumsq += (Xb ** 2).sum(dim=0)
        # cov_sum    += Zb.sum(dim=0)
        # cov_sumsq  += (Zb ** 2).sum(dim=0)
        n_cells_total += Xb.shape[0]
        
    gene_mean = gene_sum / n_cells_total
    gene_var  = gene_sumsq / n_cells_total - gene_mean ** 2
    gene_std  = torch.sqrt(torch.clamp_min(gene_var, 0.0))
    # cov_mean  = cov_sum / n_cells_total
    # cov_var   = cov_sumsq / n_cells_total - cov_mean ** 2
    # cov_std   = torch.sqrt(torch.clamp_min(cov_var, 0))  # covariates: keep tiny floor
    # return gene_mean, gene_std, cov_mean, cov_std
    return gene_mean, gene_std



def evaluate_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    recenter_y: bool = False,
    y_mean: Optional[torch.Tensor] = None,
    return_compositions: bool = False,
    method: str = "predict_ilr",  # or "softmax_celltype", "softmax_ilr"
) -> Dict[str, torch.Tensor]:
    """
    Run model on a loader and return predictions and targets aligned to dataset indices.

    Returns a dict with:
      - "preds": (S, T) predictions (ILR or composition, depending on method and flags)
      - "targets": (S, T) targets from the dataset (as provided by loader)
      - optionally "preds_comp": (S, K) compositions (if return_compositions=True and method outputs ILR)
    """
    model.eval()

    # Figure out S and T from the dataset and a sample batch
    S = len(loader.dataset)
    # Peek first batch to get output dims
    first_batch = next(iter(loader))
    # Unpack consistent with your collate: (X_cells, cell_to_batch, Z, Y_batch, sample_idx_batch)
    _, _, _, Yb0, sample_idx_batch0 = first_batch
    T = Yb0.shape[1]  # target dimensionality (e.g., K or K-1)

    preds_all = torch.empty((S, T), dtype=torch.float32)
    targets_all = torch.empty((S, T), dtype=torch.float32)

    # If you will add back y_mean (for centered ILR), move it to device once
    y_mean_dev = None
    if recenter_y and y_mean is not None:
        y_mean_dev = y_mean.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        for Xb, cell_to_batch, Zb, Yb, sample_idx_batch in loader:
            # Move tensors to device; the model will handle preprocessing
            Xb = Xb.to(device=device, dtype=getattr(model, "dtype", torch.float32), non_blocking=True)
            Zb = Zb.to(device=device, dtype=getattr(model, "dtype", torch.float32), non_blocking=True)
            cell_to_batch = cell_to_batch.to(device=device, non_blocking=True)
            sample_idx_batch = sample_idx_batch.to(device=device, non_blocking=True)
            Yb = Yb.to(device=device, dtype=torch.float32, non_blocking=True)

            # Forward pass
            preds = model(Xb, Zb, cell_to_batch, sample_idx_batch)  # shape (B, T) at sample level

            # If you centered ILR during training, add back mean here for fair comparison
            if recenter_y and y_mean_dev is not None:
                preds = preds.float() + y_mean_dev  # ensure float32 for addition

            # Move to CPU and place rows by sample_idx_batch (these are dataset-local indices 0..S-1)
            idx = sample_idx_batch.cpu().long()
            preds_all[idx] = preds.detach().cpu().float()
            targets_all[idx] = Yb.detach().cpu().float()

    out = {"preds": preds_all, "targets": targets_all}

    # Optionally return compositions if the model outputs ILR (predict_ilr or softmax_ilr)
    if return_compositions and method in ("predict_ilr", "softmax_ilr"):
        # Invert ILR to compositions
        from regression.neural_network.ilr_torch import ilr_transform_inverse
        out["preds_comp"] = ilr_transform_inverse(preds_all, dim=1)  # (S, K)
        out["targets_comp"] = ilr_transform_inverse(targets_all, dim=1)  # (S, K)

    return out


def build_datasets_from_indices(
    method: str,
    Xs_raw: List,                                  # list of S sample matrices
    Z: Union[np.ndarray, "pd.DataFrame"],          # (S, C)
    Y_imp_ilr: Optional[np.ndarray],               # (S, T_ilr) or None
    cell_type_proportions_df: Optional[Union[np.ndarray, "pd.DataFrame"]],
    indices: Dict[str, np.ndarray],                # keys: 'train', 'val' (and optionally 'test')
    dtype: torch.dtype = torch.float32,
):
    Z_np = np.asarray(Z, dtype=np.float32)

    datasets = {}
    extras = {}

    if method == "ilr_recenter":
        assert Y_imp_ilr is not None, "Y_imp_ilr must be provided for ilr_recenter."
        Y_ilr_np = np.asarray(Y_imp_ilr, dtype=np.float32)
        Y_train_mean = Y_ilr_np[indices["train"], :].mean(axis=0)
        extras["Y_train_mean"] = Y_train_mean
        Y_centered = Y_ilr_np - Y_train_mean
        for split in ("train", "val", "test"):
            if split not in indices:
                continue
            idx = indices[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_centered[idx, :],
                dtype=dtype,
            )
        return datasets, extras

    if method.endswith("ilr"):
        assert Y_imp_ilr is not None, "Y_imp_ilr must be provided for ILR methods."
        Y_ilr_np = np.asarray(Y_imp_ilr, dtype=np.float32)
        for split in ("train", "val", "test"):
            if split not in indices:
                continue
            idx = indices[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_ilr_np[idx, :],
                dtype=dtype,
            )
        extras["Y_train_mean"] = Y_ilr_np[indices["train"], :].mean(axis=0)
    else:
        assert cell_type_proportions_df is not None, "cell_type_proportions_df required for composition methods."
        Y_comp_np = np.asarray(cell_type_proportions_df, dtype=np.float32)
        for split in ("train", "val", "test"):
            if split not in indices:
                continue
            idx = indices[split]
            datasets[split] = SampleCellsDataset(
                [Xs_raw[i] for i in idx.tolist()],
                Z_np[idx, :],
                Y_comp_np[idx, :],
                dtype=dtype,
            )

    return datasets, extras



def make_kfold_train_val_test_indices(
    S: int,
    n_splits: int = 5,
    seed: int = 0,
    shuffle: bool = True,
    y_strat: Optional[np.ndarray] = None,   # shape (S,), for stratified folds
    groups: Optional[np.ndarray] = None,    # shape (S,), for grouped folds
    val_offset: int = 1                     # which fold to use as val relative to test fold
) -> List[Dict[str, np.ndarray]]:
    """
    Produce K disjoint train/val/test splits:
      - For fold f, test = folds[f], val = folds[(f + val_offset) % K], train = rest.
    All folds are disjoint; across folds, every sample appears once as test and once as val.

    Args:
      S: number of samples.
      n_splits: number of folds (must be >= 3 for non-empty train).
      seed, shuffle: passed to splitter.
      y_strat: optional labels for StratifiedKFold (mutually exclusive with groups).
      groups: optional group IDs for GroupKFold (mutually exclusive with y_strat).
      val_offset: 1 uses the next fold as validation; can be >1 if you want a gap.

    Returns:
      List of dicts, each with keys: 'train', 'val', 'test'.
    """
    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 to create disjoint train/val/test per fold.")

    indices = np.arange(S)

    # Build base folds as a list of test-index arrays (disjoint, near-equal sizes)
    if y_strat is not None and groups is not None:
        raise ValueError("Provide either y_strat (stratified) or groups (grouped), not both.")
    if y_strat is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        folds = [test_idx for _, test_idx in splitter.split(indices, y_strat)]
    elif groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        folds = [test_idx for _, test_idx in splitter.split(indices, groups=groups)]
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        folds = [test_idx for _, test_idx in splitter.split(indices)]

    # Build (train, val, test) per fold
    out: List[Dict[str, np.ndarray]] = []
    for f in range(n_splits):
        test_idx = folds[f]
        val_idx = folds[(f + val_offset) % n_splits]
        # train = all others
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i not in {f, (f + val_offset) % n_splits}])
        out.append({"train": train_idx, "val": val_idx, "test": test_idx})

    return out

import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedShuffleSplit, GroupShuffleSplit

def make_outerkfold_inner_split(
    S: int,
    n_splits: int = 5,
    seed: int = 0,
    shuffle: bool = True,
    y_strat: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    inner_val_frac_of_trainval: float = 0.10,
) -> List[Dict[str, np.ndarray]]:

    indices = np.arange(S)

    # # Outer splits (define test)
    if y_strat is not None and groups is not None:
        raise ValueError("Provide either y_strat or groups, not both.")
    if y_strat is not None:
        outer = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        outer_splits = list(outer.split(indices, y_strat))
    elif groups is not None:
        outer = GroupKFold(n_splits=n_splits)
        outer_splits = list(outer.split(indices, groups=groups))
    else:
        outer = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        outer_splits = list(outer.split(indices))

    out: List[Dict[str, np.ndarray]] = []
    for f, (trainval_idx, test_idx) in enumerate(outer_splits):
        # # Inner split (train vs val) within trainval
        if y_strat is not None:
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=inner_val_frac_of_trainval,
                random_state=seed + 10_000 + f
            )
            tr_pos, va_pos = next(inner.split(trainval_idx, y_strat[trainval_idx]))
            train_idx = trainval_idx[tr_pos]
            val_idx = trainval_idx[va_pos]

        elif groups is not None:
            inner = GroupShuffleSplit(
                n_splits=1,
                test_size=inner_val_frac_of_trainval,
                random_state=seed + 10_000 + f
            )
            tr_pos, va_pos = next(inner.split(trainval_idx, groups=groups[trainval_idx]))
            train_idx = trainval_idx[tr_pos]
            val_idx = trainval_idx[va_pos]

        else:
            rng = np.random.default_rng(seed + 10_000 + f)
            permuted = rng.permutation(trainval_idx)  # permuted VALUES, not positions
            n_val = int(round(len(trainval_idx) * inner_val_frac_of_trainval))
            val_idx = permuted[:n_val]
            train_idx = permuted[n_val:]

        out.append({"train": train_idx, "val": val_idx, "test": test_idx})

    return out