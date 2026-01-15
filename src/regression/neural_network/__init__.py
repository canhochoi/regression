from .models import *
from .trainers import (
    train_ridge_torch, run_one_seed, run_ridgecv, run_one_seed_pseudobulk_MLP, 
    train_MLP_pseudobulk, run_one_seed_MLP_pseudobulk, Trainer,
    )
from .checkfitting import CheckFitting  
from .process_data import process_data, Preprocessor, Aggregator, Postprocessor
from .ilr_torch import ilr_transform, ilr_transform_inverse
# regression/__init__.py

from .data_utils import (
    SampleCellsDataset, collate_samples_full_Z_cell, split_indices, 
    build_datasets, build_loaders, compute_cellwise_stats
    )

__all__ = ["Regressor", "PseudobulkLinearProportions", "train_ridge_torch", 
           "run_one_seed", "run_one_seed_pseudobulk_MLP", "run_ridgecv", "CheckFitting",
           "process_data", "MLP_PseudobulkLinearProportions", "train_MLP_pseudobulk",
           "run_one_seed_MLP_pseudobulk", "ilr_transform", "ilr_transform_inverse", 
           "Trainer", "Preprocessor", "Aggregator", "Postprocessor",
           "CellPredictor", "CompositionModel",
           "SampleCellsDataset", "collate_samples_full_Z_cell", "split_indices", 
            "build_datasets", "build_loaders", "compute_cellwise_stats"]

# python
# import importlib

# Public API: subpackages and convenience names
# use this when the development is done
# for developing, we can directly import from submodules
# __all__ = [
#     "models", "preprocessing",
#     "PseudobulkLinearProportions", "Regressor",
#     "train_ridge_torch", "run_one_seed", "run_ridgecv",
#     "CheckFitting"
# ]

# def __getattr__(name):
#     # Lazy-load subpackages
#     if name in ("models", "preprocessing"):
#         return importlib.import_module(f".{name}", __name__)
#     # Lazy-load classes from models
#     if name in ("PseudobulkLinearProportions", "Regressor"):
#         mod = importlib.import_module(".models", __name__)
#         return getattr(mod, name)
#     # Lazy-load trainer functions
#     if name in ("train_ridge_torch", "run_one_seed", "run_ridgecv"):
#         mod = importlib.import_module(".trainers", __name__)
#         return getattr(mod, name)
#     if name in ("CheckFitting"):
#         mod = importlib.import_module(".checkfitting", __name__)
#         return getattr(mod, name)
#     raise AttributeError(f"{__name__} has no attribute {name!r}")

# def __dir__():
#     return sorted(__all__)