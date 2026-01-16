import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Union, Optional
from skbio.stats.composition import ilr_inv, clr, ilr

from regression.neural_network.data_utils import PseudobulkAggregator, RowLibSizeNorm


# controls star imports of the module (i.e., from regression.neural_network.models import *)
__all__ = ["Regressor", "PseudobulkLinearProportions", "MLP_PseudobulkLinearProportions",
           "MLP", "CellPredictor", "CompositionModel"]

# ==========================
# Regressor (single linear layer)
# ==========================
class Regressor(nn.Module):
    '''
    A simple ridge regression model implemented as a single linear layer.
    Parameters:
    in_features: int
        Number of input features (genes, covariates).
    out_features: int
        Number of output features.
    '''
    def __init__(self, in_features: int, out_features: int, fit_intercept=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=fit_intercept)

    def forward(self, X):
        return self.linear(X)
    
class PseudobulkLinearProportions(nn.Module):
    """
    X_batch (N x G), batch_idx (N,) -> pseudobulk X_bulk (S x G) -> Linear(G->T) -> softmax -> (S x T)
    Encoder uses only linear ops (sum/mean over cells, optional row normalization, Linear).
    """
    def __init__(self, gene_dim: int, 
                num_targets: int,                 
                aggregate_mode: str = "sum",
                libsize_norm: bool = True,
                dropout = 0.1,
                scale: float = 1e6,
                scaling: bool = True,
                covariates_dim: int = 0,
                bulk_mean: torch.Tensor = None,
                bulk_std: torch.Tensor = None,
                device = "cuda",
                dtype = torch.float32
                ):
        '''
        gene_dim: number of genes
        num_targets: number of cell types
        '''
        super().__init__()
        self.gene_dim = gene_dim
        self.num_targets = num_targets
        self.dropout = dropout
        self.good_idx = None
        self.scaling = scaling
        self.covariates_dim = covariates_dim
        self.in_features = gene_dim + covariates_dim
        self.bulk_mean = bulk_mean
        self.bulk_std = bulk_std    
        self.dtype = dtype
        self.device = device

        #preprocessing
        self.aggregate = PseudobulkAggregator(mode=aggregate_mode)
        self.libnorm = RowLibSizeNorm(scale=scale) if libsize_norm else None

        #architecture
        # self.linear = nn.Linear(self.gene_dim, self.num_targets, bias=True)
        
        # self.encoder = nn.Sequential(*[self.linear, nn.Dropout(self.dropout)])
        
        self.linear = nn.Linear(self.in_features, self.num_targets, bias=False, device = self.device, dtype = self.dtype) 
        # added
        # self.log_tau = nn.Parameter(torch.log(torch.tensor(1.0)))  # learnable temperature τ (optional)


    def set_gene_mask(self, good_idx_np: np.ndarray):
        """
        Set indices of genes to keep (after pseudobulk and optional libsize normalization).
        Also reinitialize the linear layer to match the reduced input dim.
        """
        device = next(self.parameters()).device
        good_idx_t = torch.as_tensor(good_idx_np, device=device, dtype=torch.long)
        self.good_idx = good_idx_t  # buffer used in forward

        # Recreate linear with reduced input dimension
        if self.gene_dim != good_idx_t.numel():
            self.in_features = good_idx_t.numel() + self.covariates_dim
            self.linear = nn.Linear(self.in_features, self.num_targets, bias=False, device = self.device, dtype = self.dtype)

        
    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor, clamp_min: float = 1e-6):
        """
        mean, std: tensors of shape (1, G) on the same device as the module (row vectors)
        """
        # deep copy and want the copied tensor to remain connected to the original's computation graph
        self.bulk_mean = mean.clone()
        self.bulk_std = std.clone().clamp_(min=clamp_min)

    def init_bias_to_mean(self, Y_train_np, eps=1e-6):
        """
        Initialize linear.bias so that softmax(linear(0)) approximates mean proportions.
        """
        import numpy as np
        ybar = np.clip(Y_train_np.mean(axis=0), eps, 1.0)
        logits_bias = np.log(ybar)  # softmax(bias) ~ ybar when input ~ 0
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor(logits_bias, dtype=self.linear.bias.dtype, device=self.linear.bias.device))
            # Optional: zero weights for a calibrated start
            self.linear.weight.zero_()

    def forward(self, X_batch: torch.Tensor, batch_idx: torch.LongTensor, covariates: torch.Tensor = None):
    
            # Pseudobulk aggregation for each sample (linear)
            X_bulk = self.aggregate(X_batch, batch_idx)  # (S, G)
    
            # Library-size normalization and log1p
            if self.libnorm is not None:
                X_bulk = self.libnorm(X_bulk)
                
            # good_idx stores which genes are used 
            if self.good_idx is not None:
                X_bulk = X_bulk[:, self.good_idx]
                
            # inlude covariates if it's not None
            # (S, G) becomes (S, G + number of covariates)
            if covariates is not None:
                X_bulk = torch.cat((X_bulk, covariates), axis = 1) # concatenate along columns
                  
            if self.scaling == True:
                X_bulk = (X_bulk - self.bulk_mean)/self.bulk_std
            
            ilr_y = self.linear(X_bulk)
            return ilr_y, X_bulk


class MLP_PseudobulkLinearProportions(nn.Module):
    """
    X_batch (N x G), batch_idx (N,) -> pseudobulk X_bulk (S x G) -> Linear(G->T) -> softmax -> (S x T)
    Encoder uses only linear ops (sum/mean over cells, optional row normalization, Linear).
    """
    def __init__(self, gene_dim: int, 
                num_targets: int,                 
                aggregate_mode: str = "sum",
                libsize_norm: bool = True,
                dropout = 0.1,
                scale: float = 1e6,
                scaling: bool = True,
                covariates_dim: int = 0,
                bulk_mean: torch.Tensor = None,
                bulk_std: torch.Tensor = None,
                device = "cuda",
                dtype = torch.float32,
                # --- added: MLP configuration ---
                use_pre_mlp: bool = False,
                mlp_hidden_features: Union[int, Sequence[int]] = 128,
                mlp_num_hidden_layers: int = 1,
                # mlp_activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU(),
                activation: bool = False,
                mlp_activation: Optional[Union[nn.Module, Sequence[Optional[nn.Module]]]] = None,
                mlp_out_activation: Optional[nn.Module] = None,
                mlp_dropout: Optional[Union[float, Sequence[float]]] = None,
                mlp_batch_norm: bool = False,
                mlp_bias: bool = True
                ):
        '''
        gene_dim: number of genes
        num_targets: number of cell types
        '''
        super().__init__()
        self.gene_dim = gene_dim
        self.num_targets = num_targets
        self.dropout = dropout
        self.good_idx = None
        self.scaling = scaling
        self.covariates_dim = covariates_dim
        self.in_features = gene_dim + covariates_dim
        self.bulk_mean = bulk_mean
        self.bulk_std = bulk_std    
        self.dtype = dtype
        self.device = device

        #preprocessing
        self.aggregate = PseudobulkAggregator(mode=aggregate_mode)
        self.libnorm = RowLibSizeNorm(scale=scale) if libsize_norm else None

        #architecture
        # self.linear = nn.Linear(self.gene_dim, self.num_targets, bias=True)
        
        # self.encoder = nn.Sequential(*[self.linear, nn.Dropout(self.dropout)])
        
        # self.linear = nn.Linear(self.in_features, self.num_targets, bias=False, device = self.device, dtype = self.dtype) 
        # added
        # self.log_tau = nn.Parameter(torch.log(torch.tensor(1.0)))  # learnable temperature τ (optional)

        # -----------------------------------------------------------------
        # added: optional pre-aggregation MLP that maps G -> G (keeps gene alignment)
        # If you want to change dimensionality, adjust mlp_out_features AND handle masking accordingly.
        # -----------------------------------------------------------------
        self.pre_mlp = None
        if use_pre_mlp:
            self.pre_mlp = MLP(
                # in_features=self.gene_dim,
                in_features = self.in_features,
                hidden_features=mlp_hidden_features,
                # out_features=self.gene_dim,  # keep same dim so gene mask remains valid
                # out_features = self.in_features,
                out_features = self.num_targets,
                num_hidden_layers=mlp_num_hidden_layers,
                activation=activation,
                activation_type=mlp_activation,
                out_activation=mlp_out_activation,
                dropout=mlp_dropout,
                batch_norm=mlp_batch_norm,
                bias=mlp_bias,
                device=self.device,
                dtype=self.dtype,
            )

    def set_gene_mask(self, good_idx_np: np.ndarray):
        """
        Set indices of genes to keep (after pseudobulk and optional libsize normalization).
        Also reinitialize the linear layer to match the reduced input dim.
        """
        device = next(self.parameters()).device
        good_idx_t = torch.as_tensor(good_idx_np, device=device, dtype=torch.long)
        self.good_idx = good_idx_t  # buffer used in forward

        # Recreate linear with reduced input dimension
        if self.gene_dim != good_idx_t.numel():
            self.in_features = good_idx_t.numel() + self.covariates_dim
            self.linear = nn.Linear(self.in_features, self.num_targets, bias=False, device = self.device, dtype = self.dtype)

        
    def set_standardization(self, mean: torch.Tensor, std: torch.Tensor, clamp_min: float = 1e-6):
        """
        mean, std: tensors of shape (1, G) on the same device as the module (row vectors)
        """
        # deep copy and want the copied tensor to remain connected to the original's computation graph
        self.bulk_mean = mean.clone()
        self.bulk_std = std.clone().clamp_(min=clamp_min)

    def init_bias_to_mean(self, Y_train_np, eps=1e-6):
        """
        Initialize linear.bias so that softmax(linear(0)) approximates mean proportions.
        """
        import numpy as np
        ybar = np.clip(Y_train_np.mean(axis=0), eps, 1.0)
        logits_bias = np.log(ybar)  # softmax(bias) ~ ybar when input ~ 0
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor(logits_bias, dtype=self.linear.bias.dtype, device=self.linear.bias.device))
            # Optional: zero weights for a calibrated start
            self.linear.weight.zero_()

    def forward(self, X_batch: torch.Tensor, batch_idx: torch.LongTensor, covariates: torch.Tensor = None):

            # Library-size normalization and log1p
            if self.libnorm is not None:
                X_batch = self.libnorm(X_batch)
            
            X_batch = X_batch.to(device=self.device, dtype=self.dtype)

            if covariates is not None:
                X_batch = torch.cat((X_batch, covariates), dim = 1) # concatenate along columns

            # predict per-cell ilr proportions using pre-aggregation MLP
            if self.pre_mlp is not None:
                ilr_y_cell = self.pre_mlp(X_batch)

            y_cell = ilr_inv(ilr_y_cell)
            # Pseudobulk aggregation for each sample (linear)
            y_sample = self.aggregate(y_cell, batch_idx)  # (S, G)
            ilr_y = ilr(y_sample)
            return ilr_y, X_batch
    
            # -------------- original code --------------
            # inlude covariates if it's not None
            # (S, G) becomes (S, G + number of covariates)
            # if covariates is not None:
                # X_batch = torch.cat((X_batch, covariates), axis = 1) # concatenate along columns
                  
            # --- added: optional per-cell preprocessing MLP (keeps shape N x G) ---
            # if self.pre_mlp is not None:
                # X_batch = self.pre_mlp(X_batch)

            # Pseudobulk aggregation for each sample (linear)
            # X_bulk = self.aggregate(X_batch, batch_idx)  # (S, G)
                
            # inlude covariates if it's not None
            # (S, G) becomes (S, G + number of covariates)
            # if covariates is not None:
                # X_bulk = torch.cat((X_bulk, covariates), axis = 1) # concatenate along columns
                  
            
            # # good_idx stores which genes are used 
            # if self.good_idx is not None:
            #     X_bulk = X_bulk[:, self.good_idx]
                
            
            # if self.scaling == True:
            #     X_bulk = (X_bulk - self.bulk_mean)/self.bulk_std
            
            # ilr_y = self.linear(X_bulk)
            # return ilr_y, X_bulk
    

class MLP(nn.Module):
    '''
    A simple feedforward neural network with one or many hidden layers.

    Parameters:
    in_features: int
        Number of input features.
    hidden_features: int | Sequence[int]
        - If int: size of each hidden layer (replicated num_hidden_layers times).
        - If sequence: sizes of each hidden layer (e.g., [128, 256, 128]).
    out_features: int
        Number of output features.
    num_hidden_layers: int, optional
        Number of hidden layers when hidden_features is an int (default: 1).
        Ignored if hidden_features is a sequence.
    activation: nn.Module | Sequence[nn.Module]
        Activation(s) used after each hidden Linear. If a single module is provided,
        it is reused for all hidden layers. If a sequence is provided, it must match
        the number of hidden layers.
    out_activation: Optional[nn.Module]
        Optional activation after the final Linear (default: None).
    dropout: Optional[float | Sequence[float]]
        Dropout probability applied after each hidden activation. If a single float
        is provided, it is reused for all hidden layers. If a sequence is provided,
        it must match the number of hidden layers.
    batch_norm: bool
        Whether to apply BatchNorm1d to hidden layers (default: False).
    bias: bool
        Whether Linear layers use bias (default: True).
    '''
    def __init__(
        self,
        in_features: int,
        hidden_features: Union[int, Sequence[int]],
        out_features: int,
        num_hidden_layers: int = 1,
        activation: bool = False,
        # activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        activation_type: Optional[Union[nn.Module, Sequence[Optional[nn.Module]]]] = None,
        out_activation: Optional[nn.Module] = None,
        dropout: Optional[Union[float, Sequence[float]]] = None,
        batch_norm: bool = False,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        # --- Normalize hidden sizes to a list ---
        if isinstance(hidden_features, int):
            if num_hidden_layers < 0:
                raise ValueError("num_hidden_layers must be >= 0")
            hidden_sizes = [hidden_features] * num_hidden_layers
        else:
            hidden_sizes = list(hidden_features)

        num_hidden = len(hidden_sizes)

        # --- Normalize activations for hidden layers ---
        if activation:
            if isinstance(activation_type, nn.Module):
                activations = [activation_type] * num_hidden
            else:
                activations = list(activation_type)
            if len(activations) != num_hidden:
                raise ValueError("Length of activation sequence must match number of hidden layers.")

        # --- Normalize dropouts for hidden layers ---
        if dropout is None:
            dropouts = [0.0] * num_hidden
        elif isinstance(dropout, (int, float)):
            if not (0.0 <= float(dropout) < 1.0):
                raise ValueError("dropout must be in [0.0, 1.0).")
            dropouts = [float(dropout)] * num_hidden
        else:
            dropouts = [float(p) for p in dropout]
            if len(dropouts) != num_hidden:
                raise ValueError("Length of dropout sequence must match number of hidden layers.")
            if any(p < 0.0 or p >= 1.0 for p in dropouts):
                raise ValueError("All dropout probabilities must be in [0.0, 1.0).")

        # --- Build the module list ---
        layers = []
        prev = in_features

        # Hidden stack (can be empty if num_hidden == 0)
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if activation:
                layers.append(activations[i])
            if dropouts[i] > 0.0:
                layers.append(nn.Dropout(dropouts[i]))
            prev = h

        # Output layer
        layers.append(nn.Linear(prev, out_features, bias=bias))
        if out_activation is not None:
            layers.append(out_activation)

        self.network = nn.Sequential(*layers).to(device = device, dtype = dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)
    


# ----------------------------------------
# for minibatch of single cell gene expression data
# ----------------------------------------
from typing import Callable, List

ActivationFactory = Callable[[], nn.Module]

class CellPredictor(nn.Module):
    def __init__(self, in_features: int,
                 hidden_features: Union[int, Sequence[int]],
                 out_features: int,
                 num_hidden_layers: int = 1,
                 activation: bool = False,
                #  activation_type: Union[nn.Module, Sequence[Optional[nn.Module]]] = nn.ReLU(),
                 activation_type: Union[ActivationFactory, Sequence[Optional[ActivationFactory]]] = nn.ReLU,
                 out_activation: Optional[nn.Module] = None,
                 dropout: Optional[Union[float, Sequence[float]]] = None,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 bias: bool = True,
                 device: Optional[torch.device] = 'cuda',
                 dtype: Optional[torch.dtype] = torch.float32):
        super().__init__()
        hs = [hidden_features] * num_hidden_layers if isinstance(hidden_features, int) else list(hidden_features)
        # acts = ([activation_type] * len(hs) if activation and isinstance(activation_type, nn.Module)
        #         else (list(activation_type) if activation else [None] * len(hs)))
        if activation:
            if callable(activation_type):
                act_fns: List[Optional[ActivationFactory]] = [activation_type] * len(hs)
            else:
                act_fns = list(activation_type)
                if len(act_fns) != len(hs):
                    raise ValueError("activation_type list length must match number of hidden layers")
        else:
            act_fns = [None] * len(hs)

        drops = ([0.0] * len(hs) if dropout is None
                 else ([float(dropout)] * len(hs) if isinstance(dropout, (int, float))
                 else list(dropout)))
        
        layers = []
        prev = in_features
        for i, h in enumerate(hs):
            layers.append(nn.Linear(prev, h, bias=bias, dtype=dtype))
            if batch_norm: 
                layers.append(nn.BatchNorm1d(h, dtype=dtype))
            if layer_norm: 
                layers.append(nn.LayerNorm(h, dtype=dtype))
            # if acts[i] is not None: 
                # layers.append(acts[i])
            # Instantiate a fresh activation module for this layer
            if act_fns[i] is not None:
                layers.append(act_fns[i]())
            if drops[i] and drops[i] > 0.0: 
                layers.append(nn.Dropout(drops[i]))
            prev = h
        layers.append(nn.Linear(prev, out_features, bias=bias, dtype=dtype))
        if out_activation is not None: 
            layers.append(out_activation)
        self.network = nn.Sequential(*layers).to(device=device, dtype=dtype)

    def forward(self, X):
        return self.network(X)
    

from regression.neural_network.process_data import Preprocessor
from regression.neural_network.process_data import Aggregator
from regression.neural_network.process_data import Postprocessor

class CompositionModel(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 method: str = "predict_ilr",
                 aggregator_mode: str = "mean",
                 preprocessor: Preprocessor | None = None,
                 cell_predictor: CellPredictor | None = None,
                 aggregator: Aggregator | None = None,
                 postprocessor: Postprocessor | None = None,
                 device: Optional[torch.device] = 'cuda',
                 log_norm: bool = True,
                 **predictor_kwargs):
        super().__init__()
        self.method = method
        self.device = device
        self.log_norm = log_norm
        self.preprocessor = (preprocessor or Preprocessor(log_norm=self.log_norm)).to(device=self.device)
        self.aggregator = (aggregator or Aggregator(mode=aggregator_mode)).to(device=self.device)   
        self.postprocessor = (postprocessor or Postprocessor(mode="identity")).to(device=self.device)
        out_act = nn.Softmax(dim=1) if self.method in ("softmax_celltype", "softmax_ilr") else None
        self.cell_predictor = (cell_predictor or CellPredictor(
            in_features=in_features,
            out_features=out_features,
            out_activation=out_act,
            device=self.device,
            **predictor_kwargs
        )).to(device=self.device)   
        if self.method == "softmax_ilr":
            self.postprocessor.mode = "ilr"
        self.to(device=self.device)
        
    def forward(self, X, Z, cell_to_batch, sample_idx_batch):
        if Z.dim() == 2 and Z.shape[0] == sample_idx_batch.shape[0]:
            # Map sample-level covariates to cell-level
            Z = Z[cell_to_batch]    
        # convert cell_to_batch to map to sample indices in sample_idx_batch
        cell_to_batch = sample_idx_batch[cell_to_batch]

        Xp = self.preprocessor(X, Z)
        X_cell = self.cell_predictor(Xp)
        Y_sample = self.aggregator(X_cell, cell_to_batch, sample_idx_batch)
        return self.postprocessor(Y_sample)
    