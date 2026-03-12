import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import seaborn as sns

# Standard library
import os
import pickle
from typing import Literal, Optional

# Third-party
import scipy as sp
import seaborn as sns
from skbio.stats.composition import (
    closure, multiplicative_replacement, ilr, ilr_inv, clr, multi_replace
)

from joblib import Parallel, delayed

# Local imports
from run_kfold import Config, get_activation_class, load_inputs, make_outerkfold_inner_split, get_loss_class
from regression.preprocessing import build_pseudobulk_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def do_PCA(X):

    '''
    X: a dataframe

    Retunrn:
    scores_df: sample scores
    loadings_df: loading vectors 
    eig_df: eigen values dataframe
    '''
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    n_components = min(X_std.shape[0], X_std.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X_std)
    
    scores_df = pd.DataFrame(scores, index=X.index,
                             columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    
    loadings_df = pd.DataFrame(pca.components_.T, index=X.columns,
                               columns=scores_df.columns)

    # Eigenvalues (variance explained by each PC)
    eigvals = pca.explained_variance_                 # length nmax
    explained = pca.explained_variance_ratio_         # length nmax
    cum_explained = np.cumsum(explained)

    eig_df = pd.DataFrame({
                            "eigenvalue": eigvals,
                            "explained_ratio": explained,
                            "cum_explained_ratio": cum_explained
                        }, index=[f"PC{i+1}" for i in range(n_components)])
    
    return scores_df, loadings_df, eig_df

# for finding statistical significant correlation between two dataframes (e.g. PCA scores and cell type proportions)
def cross_corr(A: pd.DataFrame, B: pd.DataFrame):
    # align rows
    common = A.index.intersection(B.index)
    A = A.loc[common]
    B = B.loc[common]

    # standardize columns
    A_z = (A - A.mean(axis=0)) / (A.std(axis=0, ddof=0) + 1e-12)
    B_z = (B - B.mean(axis=0)) / (B.std(axis=0, ddof=0) + 1e-12)

    # correlation = (A_z^T B_z) / n
    C = (A_z.T @ B_z) / A_z.shape[0]
    return C


import numpy as np
import pandas as pd
from scipy import stats

# find p_values for cross-correlation between two dataframes (e.g. PCA scores and cell type proportions)
def cross_pearson_corr_pvals(A: pd.DataFrame, B: pd.DataFrame):
    common = A.index.intersection(B.index)
    A = A.loc[common]
    B = B.loc[common]

    n = A.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 samples for correlation p-values.")

    # z-score columns
    Az = (A - A.mean(0)) / (A.std(0, ddof=0) + 1e-12)
    Bz = (B - B.mean(0)) / (B.std(0, ddof=0) + 1e-12)

    # correlations (p x q)
    R = (Az.T @ Bz) / n

    # t-statistic for Pearson r
    df = n - 2
    R_clip = R.clip(-0.999999, 0.999999)
    T = R_clip * np.sqrt(df / (1.0 - R_clip**2))

    # two-sided p-values
    P = 2.0 * stats.t.sf(np.abs(T), df=df)

    R = pd.DataFrame(R, index=A.columns, columns=B.columns)
    P = pd.DataFrame(P, index=A.columns, columns=B.columns)
    return R, P


def plot_clustermap_with_annotation(
    X_df_filtered: pd.DataFrame,
    ct_annot: pd.DataFrame,
    cell_type: str,
    *,
    out_pdf: PdfPages | None = None,
    save_path: str | Path | None = None,
    # # Heatmap settings
    cmap_heatmap: str = "vlag",
    z_score: int | None = 0,
    figsize: tuple[float, float] = (16, 16),
    heat_vmin: float | None = -5,
    heat_vmax: float | None = 5,
    heat_cbar_ylabel: str = "z_score gene expression",
    heat_cbar_shift: float = 0.05,  # move heatmap colorbar up
    xtick_fontsize: int = 6,
    xtick_rotation: int = 90,
    # # Annotation strip settings
    cmap_ann: str = "Spectral_r",
    ann_rescale: str = "minmax",   # "minmax" or "raw"
    ann_vmin: float = 0.0,         # used if ann_rescale="raw"
    ann_vmax: float = 1.0,         # used if ann_rescale="raw"
    ann_cbar_pos: tuple[float, float, float, float] = (0.72, 0.965, 0.25, 0.02),
    ann_cbar_label: str = "Cell type proportion",
    ann_tick_fontsize: int = 12,
    # # clustermap layout knobs
    dendrogram_ratio=(0.12, 0.12),
    colors_ratio=0.03,
    **clustermap_kwargs
):
    """
    Plots X_df_filtered.T (genes x samples) clustermap with a single continuous annotation strip
    for `cell_type`, plus:
      - clipped heatmap color scale (heat_vmin/heat_vmax)
      - moved heatmap colorbar (heat_cbar_shift)
      - custom heatmap colorbar label
      - small xtick labels + removed x label
      - annotation-strip colorbar with correct mapping and ticks
    """

    if cell_type not in ct_annot.columns:
        raise KeyError(f"cell_type='{cell_type}' not found in ct_annot.columns")

    # # Align samples
    common = X_df_filtered.index.intersection(ct_annot.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between X_df_filtered.index and ct_annot.index")

    X = X_df_filtered.loc[common]
    v_raw = ct_annot.loc[common, cell_type].astype(float)

    # # Build annotation colors and matching norm for colorbar
    cmap_ann_obj = mpl.colormaps[cmap_ann]

    if ann_rescale == "minmax":
        vmin = float(v_raw.min())
        vmax = float(v_raw.max())
        v01 = (v_raw - vmin) / (vmax - vmin + 1e-12)
        col_colors_ct = v01.map(lambda x: mpl.colors.to_hex(cmap_ann_obj(x)))
        ann_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # for raw-value colorbar
        ann_ticks = [vmin, (vmin + vmax) / 2, vmax]
    elif ann_rescale == "raw":
        ann_norm = mpl.colors.Normalize(vmin=ann_vmin, vmax=ann_vmax)
        col_colors_ct = v_raw.map(lambda x: mpl.colors.to_hex(cmap_ann_obj(ann_norm(x))))
        ann_ticks = [ann_vmin, (ann_vmin + ann_vmax) / 2, ann_vmax]
    else:
        raise ValueError("ann_rescale must be 'minmax' or 'raw'")

    # seaborn accepts Series for col_colors (indexed by samples)
    col_colors_ct = col_colors_ct.loc[common]

    # # Plot clustermap
    g = sns.clustermap(
        X.T,
        z_score=z_score,
        cmap=cmap_heatmap,
        col_colors=col_colors_ct,
        figsize=figsize,
        vmin=heat_vmin,
        vmax=heat_vmax,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        **clustermap_kwargs
    )

    # # Move heatmap colorbar + label it
    if g.cax is not None:
        pos = g.cax.get_position()
        g.cax.set_position([pos.x0, pos.y0 + heat_cbar_shift, pos.width, pos.height])
        g.cax.set_ylabel(heat_cbar_ylabel)

    # # Tick label styling on heatmap axis
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=xtick_fontsize, rotation=xtick_rotation)
    g.ax_heatmap.set_xlabel("")

    # # Title
    g.fig.suptitle(f"{cell_type} annotation", y=1.02, fontsize=14)

    # # Add annotation-strip colorbar (matches col_colors mapping)
    sm = mpl.cm.ScalarMappable(norm=ann_norm, cmap=cmap_ann_obj)
    sm.set_array([])

    cax = g.fig.add_axes(list(ann_cbar_pos))
    cb = g.fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(ann_cbar_label)
    cb.set_ticks(ann_ticks)
    cb.ax.tick_params(labelsize=ann_tick_fontsize)

    # # Save/append
    if out_pdf is not None:
        out_pdf.savefig(g.fig, bbox_inches="tight")
        plt.close(g.fig)
    elif save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, bbox_inches="tight")
        plt.close(g.fig)
    else:
        plt.show()

    return g


def batch_clustermaps_to_one_pdf(
    X_df_filtered: pd.DataFrame,
    ct_annot: pd.DataFrame,
    cell_types: list[str],
    out_pdf_path: str | Path,
    *,
    per_celltype_dir: str | Path | None = None,
    **plot_kwargs
):
    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if per_celltype_dir is not None:
        per_celltype_dir = Path(per_celltype_dir)
        per_celltype_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf_path) as pdf:
        for ct in cell_types:
            plot_clustermap_with_annotation(
                X_df_filtered=X_df_filtered,
                ct_annot=ct_annot,
                cell_type=ct,
                out_pdf=pdf,
                **plot_kwargs
            )

            if per_celltype_dir is not None:
                plot_clustermap_with_annotation(
                    X_df_filtered=X_df_filtered,
                    ct_annot=ct_annot,
                    cell_type=ct,
                    save_path=per_celltype_dir / f"clustermap_{ct}.pdf",
                    **plot_kwargs
                )


def build_clustering_pdf(fold: int, split: str, cmap_name: str, out_pdf_path: str = "/data1/soldatr/luan/projects/cell_tissue_phenotype/scripts/deep_learning/_hand_over/plots/clustermaps.pdf", per_celltype_dir: str = "/data1/soldatr/luan/projects/cell_tissue_phenotype/scripts/deep_learning/_hand_over/plots/", method: Literal["sum_log", "log_mean"] = "sum_log") -> None:
    '''
    Build clustering heatmap and store as pdf 
    '''

    # Ensure output directories exist
    Path(out_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(per_celltype_dir).mkdir(parents=True, exist_ok=True)
    
    # this matrix is aggregating counts at cell level for each sample and log-normalize 
    Xs_pseudobulk_train = build_pseudobulk.construct(fold = fold, method = "sum_log", split = split)
    
    # this matrix is log-normalized at cell level and take the mean for each sample
    # Xs_pseudobulk_train = build_pseudobulk.construct(fold = fold, method = "log_mean", split = "train")
    
    
    X_df = pd.DataFrame(Xs_pseudobulk_train.toarray())
    ct_annot = cell_type_proportions_df.iloc[folds[fold][split]]
    # ct_annot = pd.DataFrame(cell_type_proportions_array[folds[fold]['train'], :], 
    #              index = cell_type_proportions_df.index[folds[fold]['train']],
    #              columns = cell_type_proportions_df.columns)
    #critical for alignment
    X_df.index = ct_annot.index
    X_df.columns = gnames
    
    # remove no variable genes for z-scaling
    X_df_filtered = X_df.loc[:, X_df.std(axis = 0) != 0]
    
    batch_clustermaps_to_one_pdf(
        X_df_filtered = X_df_filtered,
        ct_annot = ct_annot,
        cell_types = cell_type_proportions_df.columns.tolist(),
        out_pdf_path = out_pdf_path,
        per_celltype_dir = per_celltype_dir
        )




def batch_clustermaps_to_one_pdf_joblib(
    X_df_filtered: pd.DataFrame,
    ct_annot: pd.DataFrame,
    cell_types: list[str],
    out_pdf_path: str | Path,
    *,
    per_celltype_dir: str | Path | None = None,
    n_jobs: int = -1,
    verbose: int = 10,
    **plot_kwargs
) -> None:
    """
    Generate clustermaps with joblib parallelization.
    
    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    verbose : int, default=10
        Verbosity level for joblib progress bar
    """
    out_pdf_path = Path(out_pdf_path)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if per_celltype_dir is not None:
        per_celltype_dir = Path(per_celltype_dir)
        per_celltype_dir.mkdir(parents=True, exist_ok=True)

    # Sequential combined PDF (usually fast enough)
    print(f"Generating combined PDF with {len(cell_types)} cell types...")
    with PdfPages(out_pdf_path) as pdf:
        for ct in cell_types:
            plot_clustermap_with_annotation(
                X_df_filtered=X_df_filtered,
                ct_annot=ct_annot,
                cell_type=ct,
                out_pdf=pdf,
                **plot_kwargs
            )
    
    print(f"✓ Combined PDF saved to {out_pdf_path}")

    # Parallel individual PDFs (the slow part)
    if per_celltype_dir is not None:
        print(f"Generating individual PDFs in parallel (n_jobs={n_jobs})...")
        
        def save_single_ct(ct):
            plot_clustermap_with_annotation(
                X_df_filtered=X_df_filtered,
                ct_annot=ct_annot,
                cell_type=ct,
                save_path=per_celltype_dir / f"clustermap_{ct}.pdf",
                **plot_kwargs
            )
            return ct
        
        Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky')(
            delayed(save_single_ct)(ct) for ct in cell_types
        )
        
        print(f"✓ Individual PDFs saved to {per_celltype_dir}")
