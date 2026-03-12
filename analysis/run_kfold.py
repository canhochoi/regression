# run_kfold.py
import os
import glob
import json
import pickle
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from regression.preprocessing import CompositionalILR
from regression.neural_network.data_utils import make_outerkfold_inner_split
from regression.neural_network.parallel import worker, run_one_fold


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # # I/O
    results_dir: str = "kfold_results"
    folder_name: str = "1_layer"
    org_dir: str = "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/refined/"
    method_data: str = "max_counts"  # used for file paths

    # # Data files
    xs_pkl_path: str = "/home/lel2/luan/projects/cell_tissue_phenotype/scripts/deep_learning/Xs_samples_raw.pkl"
    meta_csv: str = "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/refined/max_counts/mean_gene_expression/metadata_samples_pseudobulk_max_counts.csv"

    # # Model/Training
    device: str = "cuda"
    dtype: str = "float32"
    aggregate_mode: str = "mean"

    # Architecture
    num_hidden_layers: int = 0               # 0 -> linear
    hidden_features: Union[int, List[int]] = 10
    activation: bool = False
    activation_type: str = "relu"            # relu, gelu, tanh, leaky_relu, ...
    dropout: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = True

    # Target / method
    method_train: str = "predict_ilr"        # your downstream "method" argument
    recenter_y: bool = False
    scaling_factor: float = 1e4
    return_compositions: bool = True

    # Optimization
    batch_size: int = 8
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    use_lr_scheduler: bool = False

    # Loss
    loss_fn: str = "mse"                      # mse, huber, l1
    loss_kwargs: Dict[str, Any] = None        # e.g. {"delta": 0.01}

    # CV
    n_splits: int = 5
    seed: int = 0
    inner_val_frac_of_trainval: float = 0.20

    def __post_init__(self):
        if self.loss_kwargs is None:
            self.loss_kwargs = {}


# ----------------------------
# Factories (strings -> torch classes)
# ----------------------------
def get_activation_class(name: str):
    name = name.lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation_type='{name}'. Choose from: {list(mapping)}")
    return mapping[name]


def get_loss_class(name: str):
    name = name.lower()
    mapping = {
        "mse": nn.MSELoss,
        "huber": nn.HuberLoss,
        "l1": nn.L1Loss,
        "mae": nn.L1Loss,
    }
    if name not in mapping:
        raise ValueError(f"Unknown loss_fn='{name}'. Choose from: {list(mapping)}")
    return mapping[name]


# ----------------------------
# Data loading
# ----------------------------
def load_inputs(cfg: Config):
    # cell type proportions
    props_csv = os.path.join(
        cfg.org_dir,
        f"{cfg.method_data}/mean_gene_expression/cell_type_proportions_pseudobulk_{cfg.method_data}.csv",
    )
    cell_type_proportions_df = pd.read_csv(props_csv, index_col=0)

    meta = pd.read_csv(cfg.meta_csv)
    assert np.all(cell_type_proportions_df.index == meta["sample_id"])

    # covariates Z: sex one-hot + age
    Z = pd.get_dummies(meta[["sex_src"]], prefix="sex_src", drop_first=True)
    Z["ages"] = meta["ages"].astype(float)
    Z = Z.astype(float)
    Z.index = meta.index

    # Xs (list of samples)
    with open(cfg.xs_pkl_path, "rb") as f:
        Xs_raw = pickle.load(f)

    # ILR target
    ilr_tx = CompositionalILR(zero_replacement=True).fit(cell_type_proportions_df.values)
    Y_imp_ilr = ilr_tx.transform(cell_type_proportions_df.values)

    return Xs_raw, Z, Y_imp_ilr, cell_type_proportions_df


# ----------------------------
# Main orchestration
# ----------------------------
def run(cfg: Config):
    # # output folder
    folder_dir = os.path.join(cfg.results_dir, cfg.folder_name)
    os.makedirs(folder_dir, exist_ok=True)

    # # derived flags
    bias = not cfg.recenter_y
    activation_cls = get_activation_class(cfg.activation_type)
    loss_cls = get_loss_class(cfg.loss_fn)

    # # load data
    Xs_raw, Z, Y_imp_ilr, cell_type_proportions_df = load_inputs(cfg)

    # # folds
    S = len(Xs_raw)
    folds = make_outerkfold_inner_split(
        S,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
        shuffle=True,
        inner_val_frac_of_trainval=cfg.inner_val_frac_of_trainval,
    )

    # # multiprocessing setup
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    n_gpus = torch.cuda.device_count()
    if cfg.device.startswith("cuda") and n_gpus == 0:
        raise RuntimeError("cfg.device is cuda but torch.cuda.device_count() == 0")

    # if CPU requested, run all folds in one "gpu_id" slot
    if not cfg.device.startswith("cuda"):
        gpu_ids = [None]
    else:
        gpu_ids = list(range(n_gpus))

    n_folds = len(folds)
    assignments = {gid: [] for gid in gpu_ids}
    for fold in range(1, n_folds + 1):
        assignments[gpu_ids[(fold - 1) % len(gpu_ids)]].append(fold)

    procs: List[mp.Process] = []
    for gpu_id in gpu_ids:
        if not assignments[gpu_id]:
            print(f"[Main] No folds assigned to device {gpu_id}, skipping")
            continue

        print(f"[Main] Starting device {gpu_id} with folds: {assignments[gpu_id]}")

        p = mp.Process(
            target=worker,
            args=(
                gpu_id,                        # your worker expects gpu_id
                assignments[gpu_id],
                Xs_raw,
                Z,
                Y_imp_ilr,
                cell_type_proportions_df,
                folds,
                cfg.hidden_features,
                cfg.num_hidden_layers,
                cfg.activation,
                activation_cls,
                cfg.dropout,
                cfg.batch_norm,
                bias,
                cfg.scaling_factor,
                cfg.method_train,
                cfg.batch_size,
                cfg.epochs,
                cfg.lr,
                cfg.weight_decay,
                cfg.return_compositions,
                folder_dir,
                cfg.layer_norm,
                loss_cls,
                cfg.loss_kwargs,
                run_one_fold,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # # aggregate summary
    json_files = glob.glob(os.path.join(cfg.results_dir, "fold_*_gpu_*.json"))
    all_mses = []
    for jf in json_files:
        with open(jf, "r") as f:
            d = json.load(f)
        if "test_mse" in d:
            all_mses.append(d["test_mse"])

    summary = {
        "config": asdict(cfg),
        "n_folds": int(len(all_mses)),
        "test_mse_mean": float(np.mean(all_mses)) if all_mses else None,
        "test_mse_std": float(np.std(all_mses)) if all_mses else None,
    }
    with open(os.path.join(cfg.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ----------------------------
# CLI (minimal)
# ----------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser()

    # # common knobs
    p.add_argument("--results_dir", default="kfold_results")
    p.add_argument("--folder_name", default="1_layer")

    p.add_argument("--method_data", default="max_counts")
    p.add_argument("--method_train", default="predict_ilr")

    p.add_argument("--num_hidden_layers", type=int, default=0)
    p.add_argument("--hidden_features", default="10")  # accepts "10" or "256,128,64"
    p.add_argument("--activation", action="store_true") #if include the flag activation, use args.activation = True, else default = False
    p.add_argument("--activation_type", default="relu")

    p.add_argument("--loss_fn", default="mse")          # mse|huber|l1
    p.add_argument("--huber_delta", type=float, default=None)

    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_norm", action="store_true")
    p.add_argument("--layer_norm", action="store_true")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--use_lr_scheduler", action="store_true")

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    # # parse hidden_features
    if "," in args.hidden_features:
        hidden_features: Union[int, List[int]] = [int(x) for x in args.hidden_features.split(",")]
    else:
        hidden_features = int(args.hidden_features)

    loss_kwargs: Dict[str, Any] = {}
    if args.loss_fn.lower() == "huber":
        if args.huber_delta is None:
            raise ValueError("--huber_delta must be provided when --loss_fn huber")
        loss_kwargs["delta"] = args.huber_delta

    cfg = Config(
        results_dir=args.results_dir,
        folder_name=args.folder_name,
        method_data=args.method_data,
        method_train=args.method_train,
        num_hidden_layers=args.num_hidden_layers,
        hidden_features=hidden_features,
        activation=args.activation,
        activation_type=args.activation_type,
        loss_fn=args.loss_fn,
        loss_kwargs=loss_kwargs,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        layer_norm=args.layer_norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_lr_scheduler=args.use_lr_scheduler,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)