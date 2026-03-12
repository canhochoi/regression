from dataclasses import dataclass
import torch
from typing import Any, Callable, Dict
import torch.nn as nn

@dataclass
class DataInputs:
    """Groups raw data inputs."""
    Xs_raw: Any  # Replace with actual type (e.g., list of tensors or DataFrame)
    Z: Any  # e.g., pd.DataFrame
    Y_imp_ilr: Any  # e.g., np.ndarray or torch.Tensor
    cell_type_proportions_df: Any  # e.g., pd.DataFrame

@dataclass
class ModelHyperparams:
    """Groups model architecture hyperparameters."""
    hidden_features: Any  # e.g., int or list[int]
    num_hidden_layers: int
    activation: bool
    dropout: Any  # e.g., float or list[float] or None
    batch_norm: bool  # Renamed from batchnorm for consistency
    bias: bool
    scaling_factor: float
    method: str
    layer_norm: bool
    loss_fn: Callable[[], nn.Module]
    loss_fn_kwargs: Dict[str, Any]
    activation_type: Callable[[], nn.Module]


@dataclass
class TrainSetup:
    """Groups training setup parameters."""
    batch_size: int
    device: torch.device
    epochs: int
    lr: float
    weight_decay: float
    return_compositions: bool = True  
    use_lr_scheduler: bool = False  