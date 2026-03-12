import torch
from torch import nn
from regression.neural_network.models import Regressor, PseudobulkLinearProportions, MLP_PseudobulkLinearProportions
from regression.neural_network.process_data   import process_data
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from typing import Callable, Optional, List
from tqdm import tqdm
import math

def run_ridgecv(Y_tr_t, Xb_tr_scaled, alphas = np.logspace(-4, 4, 20)):

    ridge = make_pipeline(
        # StandardScaler(with_mean=False, with_std=False),
        RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
    )
    ridge.fit(Xb_tr_scaled, Y_tr_t)
    
    # Extract fitted coefficients in original feature space
    # Pipeline: step 0 = scaler, step 1 = RidgeCV
    ridge_estimator = ridge.named_steps["ridgecv"]
    # When using StandardScaler + fit_intercept, scikit-learn stores coefficients
    # in the scaled space but returns them transformed back through the pipeline.
    beta_hat = ridge_estimator.coef_
    intercept_hat = ridge_estimator.intercept_
    
    # Multi-output
    y_pred = ridge.predict(Xb_tr_scaled)
    r2 = r2_score(Y_tr_t, y_pred, multioutput="uniform_average")
    # frob_err = norm(beta_hat.T - B_true, ord="fro") / np.sqrt(p * m)
    print(f"Ridge chosen alpha: {ridge_estimator.alpha_:.4g}")
    print(f"R^2 on training data (avg over outputs): {r2:.4f}")
    # print(f"Coefficient Frobenius RMSE: {frob_err:.4f}")
    return y_pred, ridge_estimator.alpha_, beta_hat

def train_ridge_torch(
    X_train, Y_train, X_val=None, Y_val=None,
    alpha=1.0, fit_intercept=False,
    optimizer="adam", lr=1e-2, epochs=1000,
    device="cuda", dtype=torch.float64, verbose=False
    ):
    
    X_train = X_train.to(device=device, dtype=dtype).contiguous()
    Y_train = Y_train.to(device=device, dtype=dtype).contiguous()
    if X_val is not None and Y_val is not None:
        X_val = X_val.to(device=device, dtype=dtype).contiguous()
        Y_val = Y_val.to(device=device, dtype=dtype).contiguous()

    n, p = X_train.shape
    t = Y_train.shape[1]
    model = Regressor(in_features=p, out_features=t, fit_intercept=fit_intercept).to(device=device, dtype=dtype)

    def ridge_objective(pred, y):
        sse = ((pred - y) ** 2).sum()
        w2 = model.linear.weight.pow(2).sum()
        return sse + alpha * w2

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=500, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(set_to_none=True)
            pred = model(X_train)
            loss = ridge_objective(pred, Y_train)
            loss.backward()
            return loss
    elif optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    with torch.no_grad():
        model.linear.weight.normal_(mean=0.0, std=1e-4)
        if fit_intercept and model.linear.bias is not None:
            model.linear.bias.zero_()

    best_val, best_state = float("inf"), None
    train_mse, val_mse = [], []

    for ep in range(epochs):
        model.train()
        if optimizer.lower() == "lbfgs":
            loss = opt.step(closure)
        else:
            opt.zero_grad(set_to_none=True)
            pred = model(X_train)
            loss = ridge_objective(pred, Y_train)
            loss.backward()
            opt.step()
        train_mse.append(loss.item())

        if X_val is not None and Y_val is not None:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val)
                loss_val = ridge_objective(pred_val, Y_val).item()
                val_mse.append(loss_val)
                if val_mse[-1] < best_val - 1e-8:
                    best_val = val_mse[-1]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((ep + 1) % 500 == 0 or ep == 1):
            msg = f"Epoch {ep+1}: train_mse={loss.item():.6e} val_mse={val_mse[-1]:.6e}" 
            if X_val is not None:
                msg += f" val_obj={val_mse:.6e}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    W = model.linear.weight.detach().cpu().numpy()
    b = None
    if fit_intercept and model.linear.bias is not None:
        b = model.linear.bias.detach().cpu().numpy()

    return {"model": model, "coef": W, "intercept": b, "train_mse": train_mse, "val_mse": val_mse}

def run_one_seed(Y_imp_ilr, Xs_raw, Z, alpha_torch = None, epochs = 10000, seed = 0, optimizer = "adam", lr = 1e-4, dtype = torch.float64, device = "cuda"):
    '''
    Run one seed of ridge regression with both scikitlearn and pytorch implementations
    Y_imp_ilr: ilr transformed Y with shape (n_samples, n_cell_types - 1)
    Xs_raw: list of gene expression matrices (cells by genes) for each sample
    Z: list of covariates (samples by covariates)
    alpha_torch: if None, use the alpha from scikitlearn ridge cv, else use the provided alpha
    epochs: number of epochs for pytorch training
    seed: random seed
    device: torch device
    optimizer: optimizer for pytorch training
    lr: learning rate for pytorch training
    dtype: torch dtype
    '''
    # store processed train, val data
    processed_dict = process_data(Y_imp_ilr, Xs_raw, Z, seed, device = device)
    Xb_tr_scaled = processed_dict['X_train_bulk']
    Y_tr_t = processed_dict['Y_tr_t']

    # run ridge cv
    y_pred, alpha, beta_hat = run_ridgecv(Y_tr_t, Xb_tr_scaled, alphas = np.logspace(-4, 4, 20))

    # test on original data
    # dtype = torch.float64
    # alpha = ridge_estimator.alpha_
    # lr = 1e-4
    if alpha_torch == None:
        alpha_torch = alpha
    # else:
        # alpha_torch = 0
    # pytorch architecture with explitict l2 regularization
    ridge_result_torch = train_ridge_torch(
        torch.tensor(Xb_tr_scaled), torch.tensor(Y_tr_t), 
        X_val= torch.tensor(processed_dict['X_val_bulk']), 
        Y_val= torch.tensor(processed_dict['Y_val_t']),
        alpha=alpha_torch, fit_intercept=False,
        optimizer=optimizer, lr=lr, epochs=epochs,
        device=device, dtype=dtype, verbose=False
    )

    # use ridge regression
    # use scikitlearn 
    ridge_reg = Ridge(alpha = alpha, fit_intercept = False, solver='sparse_cg', tol = 1e-6, random_state = seed)
    ridge_reg.fit(Xb_tr_scaled, Y_tr_t)

    # check on test data
    B_pytorch = ridge_result_torch['coef'].T
    # B_ridge = ridge_reg.coef_.T

    
    Y_pred_pytorch = processed_dict['X_test_bulk'] @ B_pytorch
    Y_pred_ridge = ridge_reg.predict(processed_dict['X_test_bulk'])    
    
    return {'pytorch_ridge': ridge_result_torch, 
            'scikitlearn_ridge': ridge_reg, 
            'processed_dict': processed_dict,
            'Y_pred_pytorch': Y_pred_pytorch,
            'Y_pred_ridge': Y_pred_ridge,
            'alpha_ridge': alpha
           }


def train_ridge_pseudobulk_MLP(
    Xb_tr, bidx_tr, Z_train, 
    Y_train, Y_val, good_genes_idx: torch.LongTensor, 
    bulk_mean: torch.Tensor, bulk_std: torch.Tensor,
    weight_decay: float = 0.0,
    Xb_val=None, bidx_val=None, Z_val=None,
    fit_intercept=False,
    optimizer="adam", lr=1e-2, epochs=1000,
    device="cuda", dtype=torch.float64, verbose=False
    ):
    
    Y_train = Y_train.to(device=device, dtype=dtype).contiguous()
    if Y_val is not None:
        Y_val = Y_val.to(device=device, dtype=dtype).contiguous()

    # build the model
    G_good = len(good_genes_idx)
    model = PseudobulkLinearProportions(
        gene_dim=Xb_tr.shape[1], #initial gene dim before gene filtering
        num_targets=Y_train.shape[1],
        aggregate_mode="sum",
        libsize_norm=True,
        scale=1e6,
        covariates_dim=Z_train.shape[1], # include covariates
        bulk_mean=bulk_mean.to(device=device, dtype=dtype),
        bulk_std=bulk_std.to(device=device, dtype=dtype),
        device = device,
        dtype = dtype
        )
    
    # set gene mask to use only good genes
    # this also rebuilds model.linear input dim
    model.set_gene_mask(good_genes_idx.cpu().numpy())          
    
    mse = nn.MSELoss()

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=500, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(set_to_none=True)
            pred = model(Xb_tr, bidx_tr, covariates=Z_train)
            loss = mse(pred, Y_train)
            loss.backward()
            return loss
    elif optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    with torch.no_grad():
        model.linear.weight.normal_(mean=0.0, std=1e-4)
        if fit_intercept and model.linear.bias is not None:
            model.linear.bias.zero_()

    best_val, best_state = float("inf"), None
    train_mse, val_mse = [], []

    for ep in range(epochs):
        model.train()
        if optimizer.lower() == "lbfgs":
            loss = opt.step(closure)
        else:
            opt.zero_grad(set_to_none=True)
            pred, _ = model(Xb_tr, bidx_tr, covariates=Z_train)
            loss = mse(pred, Y_train)
            loss.backward()
            opt.step()
        train_mse.append(loss.item())

        if Xb_val is not None and Y_val is not None:
            model.eval()
            with torch.no_grad():
                pred_val, _ = model(Xb_val, bidx_val, covariates=Z_val)
                loss_val = mse(pred_val, Y_val).item()
                val_mse.append(loss_val)
                if val_mse[-1] < best_val - 1e-8:
                    best_val = val_mse[-1]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((ep + 1) % 500 == 0 or ep == 1):
            msg = f"Epoch {ep+1}: train_mse={loss.item():.6e} val_mse={val_mse[-1]:.6e}" 
            if Xb_val is not None:
                msg += f" val_obj={val_mse:.6e}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    W = model.linear.weight.detach().cpu().numpy()
    b = None
    if fit_intercept and model.linear.bias is not None:
        b = model.linear.bias.detach().cpu().numpy()

    return {"model": model, "coef": W, "intercept": b, "train_mse": train_mse, "val_mse": val_mse}




def run_one_seed_pseudobulk_MLP(Y_imp_ilr, Xs_raw, Z, weight_decay = 0.0, epochs = 10000, seed = 0, optimizer = "adam", lr = 1e-4, dtype = torch.float64, device = "cuda"):
    '''
    Run one seed of ridge regression with both scikitlearn and pytorch implementations
    Y_imp_ilr: ilr transformed Y with shape (n_samples, n_cell_types - 1)
    Xs_raw: list of gene expression matrices (cells by genes) for each sample
    Z: list of covariates (samples by covariates)
    alpha_torch: if None, use the alpha from scikitlearn ridge cv, else use the provided alpha
    epochs: number of epochs for pytorch training
    seed: random seed
    device: torch device
    optimizer: optimizer for pytorch training
    lr: learning rate for pytorch training
    dtype: torch dtype
    '''
    # store processed train, val data
    processed_dict = process_data(Y_imp_ilr, Xs_raw, Z, seed, device = device)
    if processed_dict is None:
       while process_data is None:
           print(f"Re-running process_data with seed {seed + 1}")
           seed += 1
           processed_dict = process_data(Y_imp_ilr, Xs_raw, Z, seed, device = device) 

    # this is already filtered with good genes and scaled
    Xb_tr_scaled = processed_dict['X_train_bulk']
    Y_tr_t = processed_dict['Y_tr_t']
    Y_tr_t = torch.tensor(Y_tr_t).to(device=device, dtype=dtype).contiguous()

    good_genes_idx = processed_dict['good_genes_idx']
    bulk_mean = processed_dict['bulk_mean']
    bulk_std = processed_dict['bulk_std']    


    Y_val_t = processed_dict['Y_val_t']
    Y_val_t = torch.tensor(Y_val_t).to(device=device, dtype=dtype).contiguous()
    good_genes_idx = torch.tensor(good_genes_idx, dtype = torch.int).to(device = device)

    Xb_tr = processed_dict['Xb_tr']
    bidx_tr = processed_dict['bidx_tr']
    Z_tr = processed_dict['Z_train']
    Xb_val = processed_dict['Xb_val']
    bidx_val = processed_dict['bidx_val']
    Z_val = processed_dict['Z_val']
    


    pseudobulk_MLP_result_torch = train_ridge_pseudobulk_MLP(
                                    Xb_tr, bidx_tr, Z_tr, 
                                    Y_tr_t, Y_val_t, good_genes_idx = good_genes_idx, 
                                    bulk_mean = bulk_mean, bulk_std = bulk_std,
                                    weight_decay = weight_decay,
                                    Xb_val=Xb_val, bidx_val=bidx_val, Z_val=Z_val,
                                    fit_intercept=False,
                                    optimizer=optimizer, lr=lr, epochs=epochs,
                                    device=device, dtype=dtype, verbose=False
                                    )

    # check on test data
    B_pseudobulk_MLP = pseudobulk_MLP_result_torch['coef'].T
    # B_ridge = ridge_reg.coef_.T
    
    Y_pred_pytorch = processed_dict['X_test_bulk'] @ B_pseudobulk_MLP

    # use ridge regression
    # use scikitlearn 
    # run ridge cv
    y_pred, alpha, beta_hat = run_ridgecv(Y_tr_t.cpu().numpy(), Xb_tr_scaled, alphas = np.logspace(-4, 4, 20))
    
    ridge_reg = Ridge(alpha = alpha, fit_intercept = False, solver='sparse_cg', tol = 1e-6, random_state = seed)
    ridge_reg.fit(Xb_tr_scaled, Y_tr_t.cpu().numpy())
    
    Y_pred_ridge = ridge_reg.predict(processed_dict['X_test_bulk'])    

    return {'pseudobulk_MLP_result_torch': pseudobulk_MLP_result_torch, 
            'processed_dict': processed_dict,
            'Y_pred_pytorch': Y_pred_pytorch,
            'scikitlearn_ridge': ridge_reg, 
            'Y_pred_ridge': Y_pred_ridge,
            'alpha_ridge': alpha
           }
 


# apply MLP architecture first, then pseudobulk linear regression    
def train_MLP_pseudobulk(
    Xb_tr, bidx_tr, Z_train, 
    Y_train, Y_val = None, 
    weight_decay: float = 0.0,
    Xb_val=None, bidx_val=None, Z_val=None,
    fit_intercept=False,
    optimizer="adam", lr=1e-2, epochs=1000,
    device="cuda", dtype=torch.float64, verbose=False,
    mlp_hidden_features: int = 128,
    mlp_bias: bool = False,
    mlp_batch_norm: bool = False    
    ):
    '''
    Train MLP pseudobulk model
    Xb_tr: torch tensor list whose elements have shape (n_cells, n_genes)
    bidx_tr: torch tensor of shape (n_cells,) indicating the sample index for each cell
    Z_train: torch tensor of shape (n_samples, n_covariates)
    Y_train: torch tensor of shape (n_samples, n_cell_types - 1)
    Y_val: torch tensor of shape (n_val_samples, n_cell_types - 1)
    weight_decay: weight decay for optimizer
    Xb_val: torch tensor list whose elements have shape (n_val_cells, n_genes)
    bidx_val: torch tensor of shape (n_val_cells,) indicating the sample index for each cell
    Z_val: torch tensor of shape (n_val_samples, n_covariates)
    fit_intercept: whether to fit intercept
    optimizer: optimizer type
    lr: learning rate
    epochs: number of epochs
    device: torch device
    dtype: torch dtype
    verbose: whether to print training progress 
    '''
    Y_train = Y_train.to(device=device, dtype=dtype).contiguous()
    if Y_val is not None:
        Y_val = Y_val.to(device=device, dtype=dtype).contiguous()

    # build the model
    
    model = MLP_PseudobulkLinearProportions(gene_dim = Xb_tr.shape[1], #initial gene dim
                                            covariates_dim = Z_train.shape[1], 
                                            num_targets = Y_train.shape[1],
                                            use_pre_mlp = True,
                                            mlp_hidden_features = mlp_hidden_features,
                                            mlp_bias = mlp_bias,
                                            mlp_batch_norm = mlp_batch_norm,
                                            device = device,
                                            dtype = dtype)
      
    mse = nn.MSELoss()

    if optimizer.lower() == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=500, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(set_to_none=True)
            pred = model(Xb_tr, bidx_tr, covariates=Z_train)
            loss = mse(pred, Y_train)
            loss.backward()
            return loss
    elif optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    with torch.no_grad():
        model.linear.weight.normal_(mean=0.0, std=1e-4)
        if fit_intercept and model.linear.bias is not None:
            model.linear.bias.zero_()

    best_val, best_state = float("inf"), None
    train_mse, val_mse = [], []

    for ep in range(epochs):
        model.train()
        if optimizer.lower() == "lbfgs":
            loss = opt.step(closure)
        else:
            opt.zero_grad(set_to_none=True)
            pred, _ = model(Xb_tr, bidx_tr, covariates=Z_train)
            loss = mse(pred, Y_train)
            loss.backward()
            opt.step()
        train_mse.append(loss.item())

        if Xb_val is not None and Y_val is not None:
            model.eval()
            with torch.no_grad():
                pred_val, _ = model(Xb_val, bidx_val, covariates=Z_val)
                loss_val = mse(pred_val, Y_val).item()
                val_mse.append(loss_val)
                if val_mse[-1] < best_val - 1e-8:
                    best_val = val_mse[-1]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((ep + 1) % 500 == 0 or ep == 1):
            msg = f"Epoch {ep+1}: train_mse={loss.item():.6e} val_mse={val_mse[-1]:.6e}" 
            if Xb_val is not None:
                msg += f" val_obj={val_mse:.6e}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    W = model.linear.weight.detach().cpu().numpy()
    b = None
    if fit_intercept and model.linear.bias is not None:
        b = model.linear.bias.detach().cpu().numpy()

    return {"model": model, "coef": W, "intercept": b, "train_mse": train_mse, "val_mse": val_mse}

def run_one_seed_MLP_pseudobulk(Y_imp_ilr, Xs_raw, Z, weight_decay = 0.0, epochs = 10000, seed = 0, optimizer = "adam", lr = 1e-4, dtype = torch.float64, device = "cuda"):
    '''
    Run one seed of ridge regression with scikitlearn 
    Use results as benchmark  for comparison with MLP_pseudobulk pytorch implementation
    Y_imp_ilr: ilr transformed Y with shape (n_samples, n_cell_types - 1)
    Xs_raw: list of gene expression matrices (cells by genes) for each sample
    Z: list of covariates (samples by covariates)
    alpha_torch: if None, use the alpha from scikitlearn ridge cv, else use the provided alpha
    epochs: number of epochs for pytorch training
    seed: random seed
    device: torch device
    optimizer: optimizer for pytorch training
    lr: learning rate for pytorch training
    dtype: torch dtype
    '''
    # store processed train, val data
    processed_dict = process_data(Y_imp_ilr, Xs_raw, Z, seed, device = device)
    if processed_dict is None:
       while process_data is None:
           print(f"Re-running process_data with seed {seed + 1}")
           seed += 1
           processed_dict = process_data(Y_imp_ilr, Xs_raw, Z, seed, device = device) 

    # this is already filtered with good genes and scaled
    Xb_tr_scaled = processed_dict['X_train_bulk']
    Y_tr_t = processed_dict['Y_tr_t']
    Y_tr_t = torch.tensor(Y_tr_t).to(device=device, dtype=dtype).contiguous()

    good_genes_idx = processed_dict['good_genes_idx']
    bulk_mean = processed_dict['bulk_mean']
    bulk_std = processed_dict['bulk_std']    


    Y_val_t = processed_dict['Y_val_t']
    Y_val_t = torch.tensor(Y_val_t).to(device=device, dtype=dtype).contiguous()
    good_genes_idx = torch.tensor(good_genes_idx, dtype = torch.int).to(device = device)

    Xb_tr = processed_dict['Xb_tr']
    bidx_tr = processed_dict['bidx_tr']
    Z_tr = processed_dict['Z_train']
    Xb_val = processed_dict['Xb_val']
    bidx_val = processed_dict['bidx_val']
    Z_val = processed_dict['Z_val']
    
    # MLP_pseudobulk_result_torch = train_MLP_pseudobulk(Xb_tr, bidx_tr, Z_tr, 
    #                                                     Y_tr_t, Y_val = None, 
    #                                                     weight_decay: float = 0.0,
    #                                                     Xb_val=None, bidx_val=None, Z_val=None,
    #                                                     fit_intercept=False,
    #                                                     optimizer="adam", lr=1e-2, epochs=1000,
    #                                                     device="cuda", dtype=torch.float64, verbose=False
    #                                                     )

    MLP_pseudobulk_result_torch = train_MLP_pseudobulk(Xb_tr, bidx_tr, 
                                                       Z_tr, Y_tr_t, Y_val = Y_val_t, 
                                                       weight_decay = 1e-3, Xb_val = Xb_val, 
                                                       bidx_val = bidx_val, Z_val = Z_val, 
                                                       fit_intercept=False, optimizer="adam", lr=1e-2, epochs=1000, 
                                                       device="cuda", dtype=torch.float64, verbose=False
                                                       )
                                                   
                                                    
    

    # check on test data
    MLP_pseudobulk_model = MLP_pseudobulk_result_torch['model']
    # B_ridge = ridge_reg.coef_.T
    
    Y_pred_pytorch = MLP_pseudobulk_model(processed_dict['Xb_test'], processed_dict['bidx_test'], covariates=processed_dict['Z_test'])[0]   
    Y_pred_pytorch = Y_pred_pytorch.detach().cpu().numpy()
    # use ridge regression
    # use scikitlearn 
    # run ridge cv
    _, alpha, _ = run_ridgecv(Y_tr_t.cpu().numpy(), Xb_tr_scaled, alphas = np.logspace(-4, 4, 20))
    
    ridge_reg = Ridge(alpha = alpha, fit_intercept = False, solver='sparse_cg', tol = 1e-6, random_state = seed)
    ridge_reg.fit(Xb_tr_scaled, Y_tr_t.cpu().numpy())
    
    Y_pred_ridge = ridge_reg.predict(processed_dict['X_test_bulk'])    

    return {'MLP_pseudobulk_result_torch': MLP_pseudobulk_result_torch, 
            'processed_dict': processed_dict,
            'Y_pred_pytorch': Y_pred_pytorch,
            'scikitlearn_ridge': ridge_reg, 
            'Y_pred_ridge': Y_pred_ridge,
            'alpha_ridge': alpha
           }


# ----------------------------------------
# for minibatch of single cell gene expression data
# ----------------------------------------


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss_fn: Callable = nn.MSELoss(reduction="mean"),
                 lr: float = 1e-3,
                 weight_decay: float = 1e-3,
                 epochs: int = 500,
                 log_interval: int = 50,
                 recenter_y: bool = False,
                 rescale_y: bool = False,
                 y_mean: Optional[torch.Tensor] = None,
                 y_std: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = "cuda",
                 dtype: Optional[torch.dtype] = torch.float32,
                 verbose: bool = False,
                 # Early stopping params
                 early_stopping: bool = True,
                 patience: int = 20,
                 min_delta: float = 1e-3,
                 restore_best: bool = True,
                 min_epochs: int = 100,
                 # LR scheduler (ReduceLROnPlateau)
                 use_lr_scheduler: bool = True,
                 scheduler_name: Optional[str] = "plateau",
                 scheduler_kwargs: Optional[dict] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.log_interval = log_interval
        self.recenter_y = recenter_y
        self.rescale_y = rescale_y
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.verbose = verbose
        if optimizer is None:
            decay, no_decay = [], []
            for _, p in self.model.named_parameters():
                if not p.requires_grad: continue
                (decay if p.ndim >= 2 else no_decay).append(p)
            # set bias and LayerNorm/BatchNorm weights to no decay    
            self.optimizer = torch.optim.AdamW(
                [{"params": decay, "weight_decay": weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=lr,
            )
        else:
            self.optimizer = optimizer
        self.train_mse_history: List[float] = []
        self.val_mse_history: List[float] = []


        # Early stopping state
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_epoch: Optional[int] = None
        self.best_val: float = math.inf
        self._best_state: Optional[dict] = None
        self.min_epochs = min_epochs

        # LR scheduler
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_name = scheduler_name
        if scheduler_kwargs is None:
            scheduler_kwargs = {
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
                "threshold": 1e-6,
                "threshold_mode": "abs",
                "cooldown": 0,
                "min_lr": 1e-6
            }

        if self.use_lr_scheduler:
            match self.scheduler_name:
                case "plateau":
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
                case "cosine":
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                self.optimizer,
                                T_0=50,    # epochs per cycle (adjust)
                                T_mult=1,      # keep cycle length constant
                                eta_min=1e-5   # minimum LR at the trough
                                )

        # Move model to device/dtype once
        self.model.to(self.device)

    def _compute_y_mean(self):
        s = None
        n = 0
        for _, _, _, Yb, _ in self.train_loader:
            Yb = Yb.to(self.device, dtype=self.dtype)
            s = Yb.sum(dim=0) if s is None else s + Yb.sum(dim=0)
            n += Yb.shape[0]
        self.y_mean = s / max(n, 1)

    def _compute_y_std(self):
        # compute std of Y over training set
        if self.y_mean is None:
            self._compute_y_mean()
        s = None
        n = 0
        for _, _, _, Yb, _ in self.train_loader:
            Yb = Yb.to(self.device, dtype=self.dtype)
            diff = Yb - self.y_mean
            sq_diff = diff ** 2
            s = sq_diff.sum(dim=0) if s is None else s + sq_diff.sum(dim=0)
            n += Yb.shape[0]
        y_var = s / max(n - 1, 1)
        self.y_std = torch.sqrt(y_var)

    def train_epoch(self):
        self.model.train()
        total_sum = 0.0
        total_elems = 0
        for Xb, cell_to_batch, Zb, Yb, sample_idx_batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            Yb = Yb.to(self.device, dtype=self.dtype)
            if self.recenter_y: 
                Yb = Yb - self.y_mean
            if self.rescale_y:
                Yb = Yb / self.y_std

            Xb = Xb.to(self.device, dtype=self.dtype)
            Zb = Zb.to(self.device, dtype=self.dtype)
            cell_to_batch = cell_to_batch.to(self.device)
            sample_idx_batch = sample_idx_batch.to(self.device)
            preds = self.model(Xb, Zb, cell_to_batch, sample_idx_batch)
            if isinstance(self.loss_fn, nn.KLDivLoss):
                preds_log = torch.log(preds + 1e-8) 
                loss = self.loss_fn(preds_log.float(), Yb.float())
            else:
                loss = self.loss_fn(preds.float(), Yb.float())
            loss.backward()
            self.optimizer.step()
            # total_sum += nn.MSELoss(reduction="sum")(preds.float(), Yb.float()).item()
            # total_elems += Yb.numel()
            # average over samples (and maybe over cell types) by multiplying by batch size
            total_sum += loss.detach().item() * Yb.shape[0]
            total_elems += Yb.shape[0]

        epoch_mse = total_sum / total_elems
        self.train_mse_history.append(epoch_mse)
        return epoch_mse

    def validate(self):
        self.model.eval()
        total_sum = 0.0
        total_elems = 0
        with torch.no_grad():
            for Xb, cell_to_batch, Zb, Yb, sample_idx_batch in self.val_loader:
                Yb = Yb.to(self.device, dtype=torch.float32)
                if self.recenter_y: 
                    Yb = Yb - self.y_mean
                if self.rescale_y:
                    Yb = Yb / self.y_std

                Xb = Xb.to(self.device, dtype=self.dtype)
                Zb = Zb.to(self.device, dtype=self.dtype)
                cell_to_batch = cell_to_batch.to(self.device)
                sample_idx_batch = sample_idx_batch.to(self.device)
                preds = self.model(Xb, Zb, cell_to_batch, sample_idx_batch)
                # total_sum += nn.MSELoss(reduction="sum")(preds.float(), Yb.float()).item()
                # total_elems += Yb.numel()
                # average over samples (and maybe over cell types) by multiplying by batch size
                if isinstance(self.loss_fn, nn.KLDivLoss):
                    preds_log = torch.log(preds + 1e-8) 
                    total_sum += self.loss_fn(preds_log.float(), Yb.float()).item() * Yb.shape[0]
                else:
                    total_sum += self.loss_fn(preds.float(), Yb.float()).item() * Yb.shape[0]
                total_elems += Yb.shape[0]

        epoch_mse = total_sum / total_elems
        self.val_mse_history.append(epoch_mse)
        return epoch_mse

    def _save_best_state(self):
            # Keep a CPU copy of the best weights (free GPU memory)
            self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _maybe_early_stop(self, current_val: float, epoch: int) -> bool:
        # Returns True if we should stop early
        improved = current_val < (self.best_val - self.min_delta)
        if improved:
            self.best_val = current_val
            self.best_epoch = epoch
            if self.restore_best:
                self._save_best_state()
            return False  # do not stop
        # No improvement
        # How many epochs since best?
        epochs_since_best = epoch - (self.best_epoch if self.best_epoch is not None else -1)
        if self.early_stopping and (self.best_epoch is not None) and (epochs_since_best >= self.patience):
            # Restore best weights if requested
            if self.restore_best and self._best_state is not None:
                self.model.load_state_dict(self._best_state)
                self.model.to(self.device)
            return True
        return False

    def fit(self):
        if self.recenter_y: 
            assert self.y_mean is not None, "y_mean must be provided or computed before training when recenter_y is True"
        if self.rescale_y:
            assert self.y_std is not None, "y_std must be provided or computed before training when rescale_y is True"

        for ep in tqdm(range(self.epochs)):
            tr = self.train_epoch()
            va = self.validate()

            # LR scheduler on validation metric
            if self.use_lr_scheduler:
                if self.scheduler_name == "plateau" and np.isfinite(va):
                    self.scheduler.step(va)
                else:
                    self.scheduler.step()     
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]

            if self.verbose and ((ep + 1) % self.log_interval == 0 or ep == 0):
                print(f"Epoch {ep+1}: train_mse={tr:.3e}, val_mse={va:.3e}")
                if self.use_lr_scheduler:
                    print(f"Epoch {ep+1}: Scheduler step with val={va:.6f}. LRs: {lrs}")               

            if ep + 1 > self.min_epochs:
                # Early stopping check
                if self._maybe_early_stop(va, ep):
                    if self.verbose:
                        print(f"Early stopping at epoch {ep+1}. Best epoch was {self.best_epoch+1} with val_mse={self.best_val:.3e}")
                    break
