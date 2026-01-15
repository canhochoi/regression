from regression.neural_network import Preprocessor, CompositionModel, Trainer
from regression.neural_network.data_utils import (
    collate_samples_compact, compute_cellwise_stats, 
    evaluate_on_loader, build_datasets_from_indices
    )

from regression.neural_network import build_loaders
from regression.neural_network.dataclass_models import DataInputs, ModelHyperparams, TrainSetup

import os
from typing import Callable, Union, Any, Dict
import numpy as np
import torch
import torch.nn as nn



def worker(device_id: int,
           fold_indices_subset: list[int],
           Xs_raw: Any,
           Z: Any,
           Y_imp_ilr: Any,
           cell_type_proportions_df: Any,
           folds: list[Dict[str, Any]],
           hidden_features: Any,
           num_hidden_layers: int,
           activation: bool,
           activation_type: Callable[[], nn.Module],
           dropout: Any,
           batch_norm: bool,
           bias: bool,
           scaling_factor: float,
           method: str,
           batch_size: int,
           epochs: int,
           lr: float,
           weight_decay: float,
           return_compositions: bool,
           results_dir: str,  #in the script is called folder_dir
           layer_norm: bool,
           loss_fn: Callable[..., nn.Module],
           loss_fn_kwargs: Dict[str, Any],
           fold_runner: Callable[[DataInputs, ModelHyperparams, TrainSetup, int, Dict[str, Any]], Dict[str, Any]]
           ) -> None:
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    else:
        print(f"[WARNING] GPU {device_id} not available; falling back to CPU.")

    # Create dataclasses (shared across folds in this process)
    data = DataInputs(Xs_raw=Xs_raw,
                      Z=Z,
                      Y_imp_ilr=Y_imp_ilr,
                      cell_type_proportions_df=cell_type_proportions_df)
    hyperparams = ModelHyperparams(hidden_features=hidden_features,
                                   num_hidden_layers=num_hidden_layers,
                                   activation=activation,
                                   activation_type=activation_type, #check to store factory, not instance
                                   dropout=dropout,
                                   batch_norm=batch_norm,
                                   bias=bias,
                                   scaling_factor=scaling_factor,
                                   method=method,
                                   layer_norm=layer_norm,
                                   loss_fn=loss_fn, #check to store factory, not instance
                                   loss_fn_kwargs=loss_fn_kwargs
                                   )
    setup = TrainSetup(batch_size=batch_size,
                       device=device,
                       epochs=epochs,
                       lr=lr,
                       weight_decay=weight_decay,
                       return_compositions=return_compositions)

    for fold in fold_indices_subset:
        fold_idxs = folds[fold - 1]  # Assuming folds is 0-indexed list of dicts
        res = fold_runner(data, hyperparams, setup, fold=fold, fold_idxs=fold_idxs)

        import pickle

        # save everything 
        filename_pkl = "cuda_parallel"
        tmp_pkl = os.path.join(results_dir, f"{filename_pkl}_fold_{res['fold']}.tmp")
        with open(tmp_pkl, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_pkl, os.path.join(results_dir, f"{filename_pkl}_fold_{res['fold']}.pkl"))

        # Save test_res separately
        test_res_path = os.path.join(results_dir, f"fold_{res['fold']}_test_res.pkl")
        tmp_pkl = test_res_path + ".tmp"
        with open(tmp_pkl, "wb") as f:
            pickle.dump(res["test_res"], f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_pkl, test_res_path)


        import json
        # Save metrics for this fold
        metrics = {"fold": res["fold"], 
                   "test_mse": res["test_mse"], 
                   "train_mse_history": res["train_mse_history"],
                   "val_mse_history": res["val_mse_history"],
                   "cor_list": res["cor_list"],
                   "rsq_list": res["rsq_list"],
                   "test_res_path": test_res_path,}
        
        out_path = os.path.join(results_dir, f"fold_{res['fold']}_gpu_{device_id}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f)
        
        # Optionally save the trained model (for later prediction)
        model_path = os.path.join(results_dir, f"fold_{res['fold']}_model.pt")
        torch.save(res["trainer"].model.to("cpu"), model_path)
        res.pop("trainer")  # free up memory, remove to save as pickle
        print(f"[GPU {device_id}] Fold {fold}: test MSE={res['test_mse']:.4f}")



def run_one_fold(data: DataInputs, hyperparams: ModelHyperparams, setup: TrainSetup, fold: int, fold_idxs: Dict[str, Any]) -> Dict[str, Any]:
    '''
    DataInputs: a DataClass storing data for training
    hyperparams: a DataClass storing MLP architecture parameters
    setup: a DataClass storing parameters to run the MLP
    fold: the fold ordering (e.g., 0, 1, 2)
    fold_idxs: storing indices of samples for training, validating and testing
    '''
    
    # Unpack for clarity 
    Xs_raw = data.Xs_raw
    Z = data.Z
    Y_imp_ilr = data.Y_imp_ilr
    cell_type_proportions_df = data.cell_type_proportions_df
    
    hidden_features = hyperparams.hidden_features
    num_hidden_layers = hyperparams.num_hidden_layers
    activation = hyperparams.activation
    dropout = hyperparams.dropout
    batch_norm = hyperparams.batch_norm
    bias = hyperparams.bias
    scaling_factor = hyperparams.scaling_factor
    method = hyperparams.method
    layer_norm=hyperparams.layer_norm
    activation_type=hyperparams.activation_type #still factory
    loss_fn=hyperparams.loss_fn(**hyperparams.loss_fn_kwargs) #instantiate the factory here

    
    batch_size = setup.batch_size
    device = setup.device
    epochs = setup.epochs
    lr = setup.lr
    weight_decay = setup.weight_decay
    return_compositions = setup.return_compositions

    # Build datasets for this fold
    datasets, extras = build_datasets_from_indices(
        method=method,
        Xs_raw=Xs_raw,
        Z=Z,
        Y_imp_ilr=Y_imp_ilr,
        cell_type_proportions_df=cell_type_proportions_df,
        indices=fold_idxs,
        dtype=torch.float32,
    )
    # Build loaders (use your compact collate)
    loaders = build_loaders(datasets, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=collate_samples_compact)
    loader = loaders["train"]
    loader_val = loaders["val"]
    loader_test = loaders["test"]
    
    # Recompute training-only stats per fold
    gene_mean, gene_std = compute_cellwise_stats(loader, device="cpu")
    tol = 1e-6
    gene_mask = (gene_std > tol)
    G_kept = int(gene_mask.sum().item())
    C = np.asarray(Z).shape[1]
    in_features = G_kept + C
    out_features = (Y_imp_ilr.shape[1] if method.endswith("ilr") else np.asarray(cell_type_proportions_df).shape[1])
    age_column_id = Z.columns.tolist().index('ages')
    train_idx = fold_idxs['train']
    age_mean = Z.iloc[train_idx]['ages'].mean()
    age_std = Z.iloc[train_idx]['ages'].std()
    
    # Build preprocessor and model for this fold
    pre = Preprocessor(gene_mask=gene_mask.to(device=device),
                       gene_mean=gene_mean.to(device=device),
                       gene_std=gene_std.to(device=device),
                       cont_cov_mask=age_column_id,
                       cov_mean=torch.tensor(age_mean).to(device=device),
                       cov_std=torch.tensor(age_std).to(device=device),
                       scaling_factor=scaling_factor,
                       normalize=True)
   
    model = CompositionModel(
        in_features=in_features,
        out_features=out_features,
        method=method,
        aggregator_mode="mean",
        preprocessor=pre,
        hidden_features=hidden_features, 
        num_hidden_layers=num_hidden_layers, 
        activation=activation,
        dropout=dropout, 
        batch_norm=batch_norm, 
        bias=bias, 
        dtype=torch.float32,
        layer_norm=layer_norm,
        activation_type=activation_type,
    )
    # Train on this device
    trainer = Trainer(model, loader, loader_val,
                      lr=lr, weight_decay=weight_decay,
                      epochs=epochs, recenter_y=(method == "ilr_recenter"),
                      device=device, dtype=torch.float32, verbose=False,
                      use_lr_scheduler=True,
                      scheduler_name='cosine',
                      loss_fn = loss_fn,
                      )
    trainer.fit()
    # Evaluate on test
    from torch.nn.functional import mse_loss
    test_res = evaluate_on_loader(trainer.model, loader_test, device=trainer.device,
                                  recenter_y=trainer.recenter_y, y_mean=trainer.y_mean,
                                  return_compositions=return_compositions, method=method)
    test_mse = mse_loss(test_res["preds"], test_res["targets"], reduction="mean").item()
    
    # include checking on cell type
    cor_list = []
    rsq_list = []
    if test_res.get("preds_comp") != None and test_res.get("targets_comp") != None:
        preds_ct = test_res["preds_comp"].cpu().numpy()
        targets_ct = test_res["targets_comp"].cpu().numpy()
        cor_list = [np.corrcoef(preds_ct[:, i], targets_ct[:, i], rowvar=False)[0][1] for i in range(preds_ct.shape[1])]
        def rsquared_feature(Y_pred, Y_obs):
            SSE = np.sum((Y_pred - Y_obs)**2, axis = 0)
            SST = np.sum((Y_obs - np.mean(Y_obs, axis = 0))**2, axis = 0)
            return 1 - SSE / SST
        rsq_list = [rsquared_feature(preds_ct[:, i], targets_ct[:, i]) for i in range(preds_ct.shape[1])]

    return {"fold": fold,
            "train_mse_history": list(trainer.train_mse_history),
            "val_mse_history": list(trainer.val_mse_history),
            "test_mse": test_mse,
            "cor_list": cor_list,
            "rsq_list": rsq_list,
            "test_res": test_res,
            "trainer": trainer
            }
            






def run_one_fold_celltype(data: DataInputs, hyperparams: ModelHyperparams, setup: TrainSetup, fold: int, fold_idxs: Dict[str, Any]) -> Dict[str, Any]:
    '''
    DataInputs: a DataClass storing data for training
    hyperparams: a DataClass storing MLP architecture parameters
    setup: a DataClass storing parameters to run the MLP
    fold: the fold ordering (e.g., 0, 1, 2)
    fold_idxs: storing indices of samples for training, validating and testing
    '''
    
    # Unpack for clarity 
    Xs_raw = data.Xs_raw
    Z = data.Z
    Y_imp_ilr = data.Y_imp_ilr
    cell_type_proportions_df = data.cell_type_proportions_df
    
    hidden_features = hyperparams.hidden_features
    num_hidden_layers = hyperparams.num_hidden_layers
    activation = hyperparams.activation
    dropout = hyperparams.dropout
    batch_norm = hyperparams.batch_norm
    bias = hyperparams.bias
    scaling_factor = hyperparams.scaling_factor
    method = hyperparams.method
    layer_norm=hyperparams.layer_norm
    activation_type=hyperparams.activation_type #still factory
    loss_fn=hyperparams.loss_fn(**hyperparams.loss_fn_kwargs) #instantiate the factory here
    
    batch_size = setup.batch_size
    device = setup.device
    epochs = setup.epochs
    lr = setup.lr
    weight_decay = setup.weight_decay
    return_compositions = setup.return_compositions

    # Build datasets for this fold
    datasets, extras = build_datasets_from_indices(
        method=method,
        Xs_raw=Xs_raw,
        Z=Z,
        Y_imp_ilr=Y_imp_ilr,
        cell_type_proportions_df=cell_type_proportions_df,
        indices=fold_idxs,
        dtype=torch.float32,
    )
    # Build loaders (use your compact collate)
    loaders = build_loaders(datasets, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=collate_samples_compact)
    loader = loaders["train"]
    loader_val = loaders["val"]
    loader_test = loaders["test"]
    
    # Recompute training-only stats per fold
    gene_mean, gene_std = compute_cellwise_stats(loader, device=device)
    tol = 1e-6
    gene_mask = (gene_std > tol)
    G_kept = int(gene_mask.sum().item())
    C = np.asarray(Z).shape[1]
    in_features = G_kept + C
    out_features = (Y_imp_ilr.shape[1] if method.endswith("ilr") else np.asarray(cell_type_proportions_df).shape[1])
    age_column_id = Z.columns.tolist().index('ages')
    train_idx = fold_idxs['train']
    age_mean = Z.iloc[train_idx]['ages'].mean()
    age_std = Z.iloc[train_idx]['ages'].std()
    
    # Build preprocessor and model for this fold
    pre = Preprocessor(gene_mask=gene_mask.to(device=device),
                       gene_mean=gene_mean.to(device=device),
                       gene_std=gene_std.to(device=device),
                       cont_cov_mask=age_column_id,
                       cov_mean=torch.tensor(age_mean).to(device=device),
                       cov_std=torch.tensor(age_std).to(device=device),
                       scaling_factor=scaling_factor,
                       normalize=True)
   
    model = CompositionModel(
        in_features=in_features,
        out_features=out_features,
        method=method,
        aggregator_mode="mean",
        preprocessor=pre,
        hidden_features=hidden_features, num_hidden_layers=num_hidden_layers, activation=activation,
        dropout=dropout, batch_norm=batch_norm, bias=bias, dtype=torch.float32,
        layer_norm=layer_norm,
        activation_type=activation_type,
    )
    # Train on this device
    trainer = Trainer(model, loader, loader_val,
                      lr=lr, weight_decay=weight_decay,
                      epochs=epochs, recenter_y=(method == "ilr_recenter"),
                      device=device, dtype=torch.float32, verbose=True,
                      use_lr_scheduler=True,
                      scheduler_name='cosine',
                      loss_fn = loss_fn)
    trainer.fit()
    # Evaluate on test
    from torch.nn.functional import mse_loss
    test_res = evaluate_on_loader(trainer.model, loader_test, device=trainer.device,
                                  recenter_y=trainer.recenter_y, y_mean=trainer.y_mean,
                                  return_compositions=return_compositions, method=method)
    test_mse = mse_loss(test_res["preds"], test_res["targets"], reduction="mean").item()
    
    # include checking on cell type
    cor_list = []
    rsq_list = []
    if test_res.get("preds") != None and test_res.get("targets") != None:
        preds_ct = test_res["preds"].cpu().numpy()
        targets_ct = test_res["targets"].cpu().numpy()
        cor_list = [np.corrcoef(preds_ct[:, i], targets_ct[:, i], rowvar=False)[0][1] for i in range(preds_ct.shape[1])]
        def rsquared_feature(Y_pred, Y_obs):
            SSE = np.sum((Y_pred - Y_obs)**2, axis = 0)
            SST = np.sum((Y_obs - np.mean(Y_obs, axis = 0))**2, axis = 0)
            return 1 - SSE / SST
        rsq_list = [rsquared_feature(preds_ct[:, i], targets_ct[:, i]) for i in range(preds_ct.shape[1])]

    return {"fold": fold,
            "train_mse_history": list(trainer.train_mse_history),
            "val_mse_history": list(trainer.val_mse_history),
            "test_mse": test_mse,
            "cor_list": cor_list,
            "rsq_list": rsq_list,
            "test_res": test_res,
            "trainer": trainer
            # "model": trainer.model.cell_predictor.network
            }
