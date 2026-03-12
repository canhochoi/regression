from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import numpy as np
from skbio.stats.composition import closure, multiplicative_replacement, ilr, ilr_inv, clr
from regression.preprocessing import build_pseudobulk_matrix
from typing import List, Dict
import scipy as sp
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.utils import resample

def filter_low_variance_genes(X, gene_names=None, sd_min=1e-3):
    if sp.sparse.issparse(X):
        Xd = X.toarray().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)

    sd = Xd.std(axis=0, ddof=0)
    keep = sd >= sd_min

    if gene_names is None:
        return Xd[:, keep], keep, sd
    else:
        gene_names = np.asarray(gene_names)
        return Xd[:, keep], gene_names[keep], keep, sd



def stability_selection_multitask_enet(X, Y, B=200, sample_frac=0.8, l1_ratio=0.5, alpha=None):
    """
    If alpha is None, uses CV inside each bootstrap (slower).
    Faster approach: fix l1_ratio and alpha chosen once on full data.
    """
    p = X.shape[1]
    counts = np.zeros(p, dtype=int)

    # Standardize once globally (common in stability selection)
    # sx = StandardScaler()
    # Xs_full = sx.fit_transform(X)

    for b in range(B):
        idx = resample(np.arange(X.shape[0]), n_samples=int(sample_frac * X.shape[0]), replace=True, random_state=b)
        Xb = X[idx, :]
        Yb = Y[idx, :]

        if alpha is None:
            m = MultiTaskElasticNetCV(
                l1_ratio=[l1_ratio],
                cv=5,
                max_iter=50000,
                n_jobs=-1,
                random_state=b
            )
            m.fit(Xb, Yb)
        else:
            from sklearn.linear_model import MultiTaskElasticNet
            m = MultiTaskElasticNet(
                l1_ratio=l1_ratio,
                alpha=alpha,
                max_iter=50000,
                random_state=b
            )
            m.fit(Xb, Yb)

        # selected if any coefficient across outputs is nonzero
        selected = np.linalg.norm(m.coef_, axis=0) > 0
        counts[selected] += 1

    freq = counts / B
    return freq

def fit_and_eval_selected_genes(X_train, Y_train, X_test, selected_mask):
    Xtr = X_train[:, selected_mask]
    Xte = X_test[:, selected_mask]

    alphas = np.logspace(-6, 4, 100)
    model = RidgeCV(alphas=alphas, cv=5).fit(Xtr, Y_train)

    Y_pred = model.predict(Xte)
    return model, Y_pred
    
def run_gene_selection_per_fold(build_pseudobulk: build_pseudobulk_matrix, cell_type_proportions_df: Dict, Y_imp_ilr: np.ndarray, Z: Dict, folds: List[Dict], fold: int, genes: List[str] = None, sd_min: float = 5e-2, l1_ratio_list: List = [0.2, 0.5, 0.8, 1.0], cv: int = 5, seed: int = 0, threshold_freq: float = 0.95, threshold_freqlist: List[float] = [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]):
    '''
    folds: a list storing dictionary of train, test, split for each fold
    fold: a fold index
    genes: list of gene names 
    '''
    # this matrix is aggregating counts at cell level for each sample and log-normalize 
    Xs_pseudobulk_train = build_pseudobulk.construct(fold = fold, method = "sum_log", split = "train")
    
    
    # filter low variable genes 
    if genes is None:
        genes = [f'{i}' for i in np.arange(Xs_pseudobulk_train.shape[1])]
    Xf, genes_f, keep_mask, sd = filter_low_variance_genes(Xs_pseudobulk_train, gene_names = genes, sd_min=sd_min)
    print(Xf.shape, keep_mask.mean())
    
    
    # build design matrix
    
    # X_design = build_pseudobulk.build_X_design(X = A, Z = Z, fold = folds[fold], split = "train")
    
    #filter low variance genes 
    X_design = build_pseudobulk.build_X_design(X = Xf, Z = Z, fold = folds[fold], split = "train")
    
    
    # Prepare data
    X_train = X_design.toarray()
    Y_train = Y_imp_ilr[folds[fold]['train'], :]
    
    # 1. Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform train
    
    # check and the result is similar without gene mask
    
    # pseudobulk_gene_mask = np.linalg.norm(X_train_scaled, ord = 1, axis = 0) >= 1e-3
    
    # print(f"Number of low var genes: {(~pseudobulk_gene_mask).sum()}")
    
    # X_train_scaled = X_train_scaled[:, pseudobulk_gene_mask]
    
    # these are one-hot-encoding variables so no need to scale
    X_train_scaled[:, -Z.shape[1]:-1] = X_train[:, -Z.shape[1]:-1]
    
    # Ridge with CV to find best alpha
    alphas = np.logspace(-6, 4, 100)  # Test alphas from 0.001 to 1000
    ridge_cv = RidgeCV(alphas=alphas, cv=5)  # 5-fold CV
    ridge_cv.fit(X_train_scaled, Y_train)
    
    print(f"Best alpha: {ridge_cv.alpha_}")
    
    
    
    Xs_pseudobulk_test = build_pseudobulk.construct(fold = fold, method = "sum_log", split = "test")
    Xs_pseudobulk_test_filtered = Xs_pseudobulk_test[:, keep_mask]
    # plt.scatter((A_test @ W).ravel(), Xs_pseudobulk_test_filtered.toarray().ravel())
    
    # build design matrix
    # use sample by programs 
    # A_test = project_to_fixed_W_nnls(X = Xs_pseudobulk_test_filtered, W = W)
    # X_design_test = build_pseudobulk.build_X_design(X = A_test, Z = Z, fold = folds[fold], split = "test")
    
    X_design_test = build_pseudobulk.build_X_design(X = Xs_pseudobulk_test_filtered, Z = Z, fold = folds[fold], split = "test")
    
    
    # Prepare data
    X_test = X_design_test.toarray()
    Y_test = Y_imp_ilr[folds[fold]['test'], :]
    
    X_test_scaled = scaler.transform(X_test)  # Use sd from train for test sets
    
    # these are one-hot-encoding variables so no need to scale
    X_test_scaled[:, -Z.shape[1]:-1] = X_test[:, -Z.shape[1]:-1]
    
    Y_test = Y_imp_ilr[folds[fold]['test'], :]
    Y_pred_test = ridge_cv.predict(X_test_scaled)
    
    print(f"Train R²: {ridge_cv.score(X_train_scaled, Y_train):.4f}")
    print(f"Test R²: {ridge_cv.score(X_test_scaled, Y_test):.4f}")
    
    
    
    # ncols = 4
    # nrows = round(Y_test.shape[1]/ncols) + 1
    
    ct_preds = ilr_inv(Y_pred_test)
    ct_obs = ilr_inv(Y_test)
    
    # # ct_cor_regression = [np.corrcoef(ct_preds[:, i], ct_obs[:, i])[0][1] for i in range(cell_type_proportions_df.shape[1])]
    
    ct_cor_regression_org = [sp.stats.spearmanr(ct_preds[:, i], ct_obs[:, i])[0] for i in range(cell_type_proportions_df.shape[1])]

    # try sparse multi-output model
    
    model = MultiTaskElasticNetCV(
            l1_ratio=l1_ratio_list,
            cv=cv,
            max_iter=10000,
            n_jobs=-1,
            random_state=seed
        )
    
    model.fit(X_train_scaled, Y_train)

    alpha_star = model.alpha_
    l1_ratio_star = model.l1_ratio_
    
    print(f"Optimal alpha: {alpha_star}")
    print(f"Optimal l1 ratio: {l1_ratio_star}")

    # gene selection based on frequency

    freq = stability_selection_multitask_enet(
                                                X_train_scaled, Y_train, 
                                                B=200,
                                                sample_frac=0.8,
                                                l1_ratio=l1_ratio_star,
                                                alpha=alpha_star
                                                )
    
    freq_mask = freq>threshold_freq
    print(f"Numer of genes > {threshold_freq}: {freq_mask.sum()}")

    
    ct_cor_regression_list = []
    for threshold_freq in threshold_freqlist:
        freq_mask = freq>threshold_freq
        print(f"Numer of genes > {threshold_freq}: {freq_mask.sum()}")
        
        while freq_mask.sum() == 0:
            print("No genes satisfy threshold! Reducing threshold")
            threshold_freq = threshold_freq - 0.01
            if threshold_freq == 0.95:
                break
            freq_mask = freq>threshold_freq

        model_selected_genes, Y_pred_selected_genes = fit_and_eval_selected_genes(X_train_scaled, Y_train, 
                                                                                  X_test_scaled, 
                                                                                  selected_mask = freq_mask)
        
        ct_preds = ilr_inv(Y_pred_selected_genes)
        ct_obs = ilr_inv(Y_test)
        ct_cor_regression = [sp.stats.spearmanr(ct_preds[:, i], ct_obs[:, i])[0] for i in range(cell_type_proportions_df.shape[1])]
        ct_cor_regression_list.append(ct_cor_regression)

    ct_cor_regression_list = np.array(ct_cor_regression_list)
    
    return {'fold': fold,
            'keep_mask': keep_mask, 
            'freq': freq,
            'genes_name_keep_mask': genes_f,
            'ct_cor_regression_org': ct_cor_regression_org,
            'Y_test': Y_test,
            'Y_pred_test': Y_pred_test,
            'ct_cor_regression_list': ct_cor_regression_list
           }


