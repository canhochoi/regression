from typing import Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from regression.neural_network.data_utils import PseudobulkAggregator, RowLibSizeNorm

def make_batch_from_list_sparse(X_list, device="cpu", dtype=torch.float32):
    """
    Concatenate per-sample matrices row-wise, convert to dense torch tensor, and build batch_idx.
    WARNING: this densifies; ensure memory can hold sum(n_cells) x n_genes for the list.
    """
    tensors = []
    idxs = []
    for i, Xi in enumerate(X_list):
        if sp.issparse(Xi):
            Xi_np = Xi.toarray().astype(np.float32, copy=False)
        elif isinstance(Xi, np.ndarray):
            Xi_np = Xi.astype(np.float32, copy=False)
        else:
            # assume already torch tensor
            Xi_np = Xi
        t = Xi_np if torch.is_tensor(Xi_np) else torch.tensor(Xi_np, dtype=dtype)
        tensors.append(t)
        idxs.append(torch.full((t.shape[0],), i, dtype=torch.long))
    X_batch = torch.cat(tensors, dim=0).to(device)
    batch_idx = torch.cat(idxs, dim=0).to(device)
    return X_batch, batch_idx

def stratified_split_indices(S, test_frac = 0.2, val_frac = 0.1, seed = 0):
    '''
    S: length of samples 
    '''
    all_idx = np.arange(S)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(S)
    n_test = int(round(test_frac * S))
    n_val = int(round(val_frac * S))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return train_idx, val_idx, test_idx

def subset_arr(A, idx): 
    return A[idx]

def matrix_splitting(Y, test_frac = 0.2, val_frac = 0.1, seed = 0):
    '''
    Y is a 2D numpy array that is ilr-transformed from cell type proportions 
    '''
    # indices for training at sample level
    train_idx_sample, val_idx_sample, test_idx_sample = stratified_split_indices(Y.shape[0], test_frac = test_frac, val_frac = val_frac, seed = seed)
    Y_train = subset_arr(Y, train_idx_sample)
    Y_val = subset_arr(Y, val_idx_sample)
    Y_test = subset_arr(Y, test_idx_sample)
    # return Y_train, Y_test, Y_val
    return {
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test,
        "train_idx": train_idx_sample, 
        "val_idx": val_idx_sample, 
        "test_idx": test_idx_sample,
    }

def subset_list(Xs, idx_list):
    X_list = []
    for i, (X, id) in enumerate(zip(Xs, idx_list)):
        X_list.append(subset_arr(X, id))
    return X_list
    
def gene_expression_list_splitting(Xs, test_frac = 0.2, val_frac = 0.1, seed = 0):
    # indices to get from each single cell matrices to do pseudobulk
    train_idx, val_idx, test_idx = [], [], []
    for i in np.arange(len(Xs)):
        # subset cells in each gene expression matrix
        train_idx_sample, val_idx_sample, test_idx_sample = stratified_split_indices(Xs[i].shape[0], test_frac = test_frac, val_frac = val_frac, seed = seed)
        # insert list as elements into a list
        train_idx.append(train_idx_sample)
        val_idx.append(val_idx_sample)
        test_idx.append(test_idx_sample)

    X_train = subset_list(Xs, train_idx)
    X_val = subset_list(Xs, val_idx)
    X_test = subset_list(Xs, test_idx)

    return X_train, X_val, X_test


def process_data(Y_imp_ilr, Xs_raw, Z, seed, device):
    '''
    Y_imp_ilr: numpy array of shape (n_samples, n_cell_types - 1), ilr transformed cell type proportions
    Xs_raw: list of gene expression matrices (cells by genes) for each sample
    Z: list of covariates (samples by covariates)
    seed: random seed for splitting
    device: torch device
    '''
    # containing indices for training, testing and validating 
    ilr_samples_split = matrix_splitting(Y_imp_ilr, seed = seed)
    Y_train = ilr_samples_split["Y_train"]
    Y_test = ilr_samples_split["Y_test"]
    Y_val = ilr_samples_split["Y_val"]

    # use all cells to calculate pseudobulk, store them in X_train
    X_train, _, _ = gene_expression_list_splitting(Xs_raw, test_frac = 0, val_frac = 0, seed = seed)
    # select samples based on X_train because it is used for pseudobulk
    X_val = [X_train[i] for i in ilr_samples_split["val_idx"]]
    X_test = [X_train[i] for i in ilr_samples_split["test_idx"]]
    X_train = [X_train[i] for i in ilr_samples_split["train_idx"]]
    
    # for using subset of cells in each gene expression matrix to do pseudobulk
    # test_frac and val_frac just for subsetting here 
    
    # X_train, _, _ = gene_expression_list_splitting(Xs_raw, test_frac = 0.1, val_frac = 0.1, seed = seed)
    # select samples based on X_train because it is used for pseudobulk
    # X_val = [X_train[i] for i in ilr_samples_split["val_idx"]]
    # X_test = [X_train[i] for i in ilr_samples_split["test_idx"]]
    # X_train = [X_train[i] for i in ilr_samples_split["train_idx"]]

    # ids of samples for trainig, testing, validating
    sample_train = torch.from_numpy(ilr_samples_split["train_idx"]).to(device)
    sample_test = torch.from_numpy(ilr_samples_split["test_idx"]).to(device)
    sample_val = torch.from_numpy(ilr_samples_split["val_idx"]).to(device)
    
    Xb_tr, bidx_tr = make_batch_from_list_sparse(X_train, device=device)
    Xb_val, bidx_val = make_batch_from_list_sparse(X_val, device=device)
    Xb_test, bidx_test = make_batch_from_list_sparse(X_test, device=device)

    # sum counts 
    X_train_bulk = PseudobulkAggregator()(Xb_tr, bidx_tr)
    # normalize lib size and log1p
    X_train_bulk = RowLibSizeNorm()(X_train_bulk)
    
    good_genes_idx = torch.where(X_train_bulk.var(axis = 0) != 0)[0]
    bulk_mean = X_train_bulk.mean(axis = 0, keepdim = True)[:, good_genes_idx]
    bulk_std = X_train_bulk.std(axis = 0, keepdim = True)[:, good_genes_idx]

    # build the model with covariates
    # for metadata
    Z_val = torch.from_numpy(np.array(Z)[ilr_samples_split["val_idx"],:]).to(device = device, dtype = torch.float32)
    Z_test = torch.from_numpy(np.array(Z)[ilr_samples_split["test_idx"],:]).to(device = device, dtype = torch.float32)
    Z_train = torch.from_numpy(np.array(Z)[ilr_samples_split["train_idx"],:]).to(device = device, dtype = torch.float32)

    Z_train_mean = Z_train.mean(axis = 0, keepdims = True)
    Z_train_std = Z_train.std(axis = 0, keepdims = True)
    
    if torch.any(Z_train_std == 0):
        print(f"Change seeds because std of covariates {torch.where(Z_train_std == 0)[0].cpu().numpy()} is 0")
        return None
    else:
        bulk_mean = torch.cat((bulk_mean, Z_train_mean), axis = 1)
        bulk_std = torch.cat((bulk_std, Z_train_std), axis = 1)
        
        
        Y_tr_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

        # get the mean of Y_tr_t
        tr_t_mean = Y_tr_t.mean(axis = 0, keepdim = True)
        # center
        Y_tr_t = Y_tr_t - tr_t_mean
        Y_val_t = Y_val_t - tr_t_mean
        Y_test_t = Y_test_t - tr_t_mean

    
        Y_tr_t = Y_tr_t.detach().cpu().numpy()
        Y_val_t = Y_val_t.detach().cpu().numpy()
        Y_test_t = Y_test_t.detach().cpu().numpy()

        def center_scale_pytorch_to_numpy(X,Z,good_genes_idx,XZ_mean,XZ_std):
            XZ = torch.cat((X[:, good_genes_idx], Z), axis = 1)
            XZ = (XZ - XZ_mean) / XZ_std
            return XZ.detach().cpu().numpy()
            
        # X_train_bulk = torch.cat((X_train_bulk, Z_train), axis = 1)
        # X_train_bulk = (X_train_bulk - bulk_mean) / bulk_std
        # X_train_bulk = X_train_bulk.detach().cpu().numpy()
    
        X_train_bulk = center_scale_pytorch_to_numpy(X_train_bulk, Z_train, good_genes_idx, bulk_mean, bulk_std)
        
        # sum counts 
        X_val_bulk = PseudobulkAggregator()(Xb_val, bidx_val)
        # normalize lib size and log1p
        X_val_bulk = RowLibSizeNorm()(X_val_bulk)
        X_val_bulk = center_scale_pytorch_to_numpy(X_val_bulk, Z_val, good_genes_idx, bulk_mean, bulk_std)


        # sum counts 
        X_test_bulk = PseudobulkAggregator()(Xb_test, bidx_test)
        # normalize lib size and log1p
        X_test_bulk = RowLibSizeNorm()(X_test_bulk)
        X_test_bulk = center_scale_pytorch_to_numpy(X_test_bulk, Z_test, good_genes_idx, bulk_mean, bulk_std)

        
        # val_loss = np.mean(ridge_reg.predict(X_val_bulk) - Y_val_t)**2                               
        return {'X_train_bulk': X_train_bulk, 
                'Y_tr_t': Y_tr_t, 
                'X_val_bulk': X_val_bulk,
                'Y_val_t': Y_val_t,
                'X_test_bulk': X_test_bulk,
                'Y_test_t': Y_test_t,
                'tr_t_mean': tr_t_mean,
                'good_genes_idx': good_genes_idx.cpu().numpy(),
                'bulk_mean': bulk_mean,
                'bulk_std': bulk_std,
                'Xb_tr': Xb_tr,
                'bidx_tr': bidx_tr,
                'Xb_val': Xb_val,
                'bidx_val': bidx_val,
                'Xb_test': Xb_test,
                'bidx_test': bidx_test,
                'Z_train': Z_train,
                'Z_val': Z_val,
                'Z_test': Z_test
                }
    


# ----------------------------------------
# for minibatch of single cell gene expression data
# ----------------------------------------


class Preprocessor(nn.Module):
    def __init__(self, gene_mask: Optional[torch.Tensor] = None,
                 gene_mean: Optional[torch.Tensor] = None,
                 gene_std: Optional[torch.Tensor] = None,
                 cont_cov_mask: Optional[torch.Tensor] = None,
                 cov_mean: Optional[torch.Tensor] = None,
                 cov_std: Optional[torch.Tensor] = None,
                 log_norm: bool = True,
                 scaling_factor: float = 1e4,
                 normalize: bool = True):
        '''
        gene_mask: mask for genes to be used
        gene_mean: mean for genes to be normalized
        gene_std: std for genes to be normalized
        scaling_factor: scaling factor for library size normalization
        normalize: whether to do normalization  
        cont_cov_mask: column id for continuous covariates in Z
        cov_mean: mean for continuous covariates to be normalized
        cov_std: std for continuous covariates to be normalized
        '''
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene_mask = gene_mask
        self.cont_cov_mask = cont_cov_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.log_norm = log_norm
        
        if gene_mean is not None: 
            self.register_buffer("gene_mean", gene_mean[gene_mask].to(device=device))
        else: 
            self.gene_mean = None
        if gene_std is not None: 
            self.register_buffer("gene_std", gene_std[gene_mask].to(device=device))
        else: 
            self.gene_std = None
        if cov_mean is not None: 
            self.register_buffer("cov_mean", cov_mean)
        else: 
            self.cov_mean = None
        if cov_std is not None: 
            self.register_buffer("cov_std", cov_std)
        else: 
            self.cov_std = None
        

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            
            if self.log_norm:
                denom = X.sum(dim=1, keepdim=True)
                X = torch.log1p(X / denom * self.scaling_factor)
            else:
                pass 

            if self.gene_mask is not None:
                X = X[:, self.gene_mask]
            if (self.gene_mean is not None) and (self.gene_std is not None):
                X = (X - self.gene_mean) / self.gene_std
            if (self.cont_cov_mask is not None) and (self.cov_mean is not None) and (self.cov_std is not None):
                Z_scaled = Z.clone()
                Z_cont = Z[:, self.cont_cov_mask]
                Z_scaled[:, self.cont_cov_mask] = (Z_cont - self.cov_mean) / self.cov_std
            else:
                Z_scaled = Z
        else:
            Z_scaled = Z
        return torch.cat((X, Z_scaled.to(dtype=X.dtype)), dim=1)
    

class Aggregator(nn.Module):
    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ("sum", "mean")
        self.mode = mode

    def forward(self, X_cells: torch.Tensor, cell_to_batch: torch.LongTensor, sample_idx_batch: torch.Tensor) -> torch.Tensor:
        B = sample_idx_batch.shape[0]
        D = X_cells.shape[1]
        # Local IDs fast path
        if torch.max(cell_to_batch) < B and torch.min(cell_to_batch) >= 0:
            out = torch.zeros((B, D), device=X_cells.device, dtype=X_cells.dtype)
            out.index_add_(0, cell_to_batch, X_cells)
            if self.mode == "mean":
                counts = torch.bincount(cell_to_batch, minlength=B).clamp_min(1).to(out.dtype).unsqueeze(1)
                out = out / counts
            return out
        # Remap dataset IDs
        ids_sorted, order = torch.sort(sample_idx_batch)
        
        # get index of batch ids for each cell based on ids_sorted
        # bucketize gives right hand side index which is similar to index of samples in ids_sorted
        # for the default case right = False

        pos_sorted = torch.bucketize(cell_to_batch, ids_sorted, right=False)
        ok = ids_sorted[pos_sorted] == cell_to_batch
        if not ok.all():
            bad = cell_to_batch[~ok]
            raise ValueError(f"cell_to_batch contains IDs not in sample_idx_batch: {bad.tolist()}")
        out_sorted = torch.zeros((B, D), device=X_cells.device, dtype=X_cells.dtype)
        out_sorted.index_add_(0, pos_sorted, X_cells)
        inv_order = torch.empty_like(order)

        # inv_order stores indices in ids_sorted for each element in the ids
        # e.g., ids = [10, 5, 7], ids_sorted = [5, 7, 10], order = [1, 2, 0]
        # inv_order = [2, 0, 1]
        # ids = ids_sorted[inv_order]

        inv_order[order] = torch.arange(B, device=order.device)
        out = out_sorted[inv_order]
        if self.mode == "mean":
            counts_sorted = torch.bincount(pos_sorted, minlength=B).clamp_min(1).to(out.dtype).unsqueeze(1)
            out = out / counts_sorted[inv_order]
        return out


from regression.neural_network.ilr_torch import ilr_transform

class Postprocessor(nn.Module):
    def __init__(self, mode: str = "identity", ilr_dim: int = 1):
        super().__init__()
        self.mode = mode
        self.ilr_dim = ilr_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return X
        if self.mode == "ilr":
            return ilr_transform(X, dim=self.ilr_dim)
        raise ValueError(f"Unknown postprocessor mode: {self.mode}")


