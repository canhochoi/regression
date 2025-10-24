# this is the one for the python script
# sampling for two or more cell types

import argparse
import scanpy as sc
import fastools as ft
import numpy as np
import rapids_singlecell as rsc
import numpy as np
import jax
import torch
import jax.numpy as jnp 
import os
import re

import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.sparse import issparse

def filter_method(adata = None, method = ["max_counts", "min_cells"], min_cells = 3):
    if method == "max_counts":
        return filter_method_max_counts(adata)
    elif method == "min_cells":
        return filter_method_min_cells(adata, min_cells = min_cells)
    else:
        raise ValueError(f"Unknown method={method!r}.")
    
    
def filter_method_max_counts(adata):
    # filter genes less than 3 counts
    max_counts_per_gene = np.max(adata.X, axis=0)
    
    max_counts_per_gene = max_counts_per_gene.toarray().flatten()
        
    gene_mask = max_counts_per_gene >= 3
    
    print(f'Number of genes to keep = {gene_mask.sum()}')
    
    # subset AnnData object
    return adata[:, gene_mask].copy()

def filter_method_min_cells(adata, min_cells = 3):
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.filter_genes(adata, min_cells = min_cells)
    rsc.get.anndata_to_CPU(adata)
    return adata.copy()

def normalize_cell_types(arg):
    # Accept str (possibly comma/space separated) or list/tuple
    if isinstance(arg, str):
        # Split on commas or whitespace, strip quotes
        parts = [s.strip(" '\"") for s in re.split(r"[,\s]+", arg) if s.strip()]
        return parts if parts else []
    elif isinstance(arg, (list, tuple)):
        return [str(s).strip(" '\"") for s in arg]
    else:
        raise TypeError(f"Unsupported type for cell_type: {type(arg)}")
            
    
def subset_adata_per_sample_many_cell_types(our_cdata, cell_type_list, n_cells=500, seed=None):
    '''
    cell_type_list is a list of cell types (e.g., ['HSC_MPP', 'B']) 
    '''
    
    if seed is not None:
        rng = np.random.default_rng(seed)

    cell_combined_samples = []

    # Group by sample_id once (faster than repeated filtering)
    sample_groups = our_cdata.obs.groupby('sample_id')
    
    for sample_id, group in sample_groups:
        
        # all cells in sample
        sample_cells = group.index  
        
        picked_cells_list =[]
        type_cells_list = []

        for cell_type in cell_type_list:

            if isinstance(cell_type, str):
                if cell_type == "HSC_MPP":
                    cell_type = cell_type.split("_")
                else:
                    cell_type = [cell_type]
                    
            # target cell types inside this sample
            type_cells = group[group['cell_type'].isin(cell_type)].index
            # store all cells subject to sampling in a list 
            type_cells_list.extend(type_cells.tolist())
    
            # sample if necessary
            if len(type_cells) > n_cells:
                picked_cells = rng.choice(type_cells, n_cells, replace=False)
            else:
                picked_cells = type_cells
                
            # combine all sampled cells    
            picked_cells_list.extend(picked_cells.tolist())

        # get remaining non-target cells 
        other_cells = sample_cells.difference(pd.Index(type_cells_list))
           
        #store cells to be kept    
        cell_combined_samples.extend(other_cells.tolist())
        cell_combined_samples.extend(picked_cells_list)

    return our_cdata[cell_combined_samples, :].copy()
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    choices=["max_counts", "min_cells"],
    default="min_cells",
    help="Filtering method to use: 'max_counts' or 'min_cells' (default: min_cells)"
    )
parser.add_argument(
    "--cell_type_list",
    type=str,
    nargs='+',
    required=True,
    help="Celltypes to calculate, space-separated (e.g., 'HSC_MPP', 'B')" #bash will produce args.cell_type_list = ['HSC_MPP', 'B']
    )
parser.add_argument(
    "--path_dir",
    type=str,
    help="Folder to store results (e.g., ../results/normal_HSC/pseudobulk/)"
    )
parser.add_argument(
    "--n_sampling",
    type=int,
    help="Number of times to do sampling"
    )
parser.add_argument(
    "--ncells_sampling",
    type=int,
    help="Number of cells to do sampling"
    )
# Parse arguments
args = parser.parse_args()

# # to run in jupyter

# from argparse import Namespace

# args = Namespace(
#         # method = "min_cells",
#         method = "max_counts", 
#         cell_type_list = ['HSC_MPP', 'B'],
#         # cell_type = "B",
#         # path_dir = "/home/lel2/luan/projects/cell_tissue_phenotype/running_slurm/test/",
#         path_dir = "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/subsampling/",
#         n_sampling = 2,
#         ncells_sampling = 500
#         )


# use harmony-corrected, 5000 top variable genes 
# our_cdata = sc.read("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/our_cdata_5000.h5ad")

# use top 5000 variable genes, raw counts with the paper annotation
# our_cdata = sc.read("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/our_cdata_raw_counts_5000.h5ad")

# use all cells with raw counts
our_cdata = sc.read("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/our_cdata_raw_celltypes.h5ad")

# filter genes or cells 
our_cdata_org = filter_method(adata = our_cdata, method = args.method)

# remove some of the cells with abnormal read counts
gene_counts_per_cell = our_cdata_org.X.sum(axis = 1).A1
q_low, q_high = np.percentile(gene_counts_per_cell, [5, 95])
print(f"5 percentile is {q_low}, 95 percentile is {q_high}")
our_cdata_org = our_cdata_org[(gene_counts_per_cell > q_low) & (gene_counts_per_cell < q_high)].copy()

df_covariate = pd.read_csv("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_covariates.csv")

#get sample_id
our_cdata_org.obs = (
    our_cdata_org.obs.reset_index() # index (cell barcodes) becomes a column called 'index'
    .merge(df_covariate, on = "indiv_id", how = "left") #merge
    .set_index("index") # restore original row names as index
    )
our_cdata_org.obs.index.name = None
our_cdata_org.obs.head()

#combined sexes and src into one column
our_cdata_org.obs['sex_src'] = our_cdata_org.obs['sexes'].astype(str) + '_' + our_cdata_org.obs['src'].astype(str)
our_cdata_org.obs['sex_src'] = our_cdata_org.obs['sex_src'].astype("category")

# saving counts data
our_cdata_org.layers['counts'] = our_cdata_org.X.copy()
our_cdata_org.layers['counts'].data

# create sample_id 
our_cdata_org.obs['sample_id'] = our_cdata_org.obs['exp_name'].astype('string').str.cat(our_cdata_org.obs['indiv_id'].astype('string'), sep = '_')
our_cdata_org.obs['sample_id'] = our_cdata_org.obs['sample_id'].astype('category')

file_dir = os.path.join(args.path_dir,'_'.join(args.cell_type_list))
os.makedirs(file_dir, exist_ok = True)

for sampling_id in np.arange(args.n_sampling):
    print(f"Doing sampling {sampling_id}")
    # subset AnnData
    # if args.cell_type == "HSC_MPP":
    #     our_cdata = subset_adata_per_sample_fast(our_cdata_org, cell_type=args.cell_type, n_cells=100, seed=sampling_id)
    # else:
    #     our_cdata = subset_adata_per_sample_fast(our_cdata_org, cell_type=args.cell_type, n_cells=100, seed=sampling_id)
    
    our_cdata = subset_adata_per_sample_many_cell_types(our_cdata_org, args.cell_type_list, n_cells = args.ncells_sampling, seed=sampling_id)
    
    
    # Group by samples and cell type, then count
    cell_type_counts = our_cdata.obs.groupby(['sample_id', 'cell_type']).size().unstack(fill_value=0)
    cell_type_proportions = cell_type_counts.div(cell_type_counts.sum(axis = 1), axis = 0)
    cell_type_proportions
    
    # get meta data for cell_type_proportions
    
    sex_map = (our_cdata.obs
               .drop_duplicates('sample_id')
               .set_index('sample_id')[['sex_src', 'ages']])
    
    # meta = sex_map.reindex(log_cpm_df.index).to_frame(name='sex_src')
    meta = sex_map.reindex(cell_type_proportions.index)
    meta['sex_src'] = meta['sex_src'].astype('category')
    meta['ages'] = meta['ages'].astype('int')
    meta

    # saving for combined cell types
    meta.to_csv(os.path.join(file_dir, f"metadata_samples_pseudobulk_{args.method}_{sampling_id}.csv"))
    cell_type_counts.to_csv(os.path.join(file_dir, f"cell_type_counts_pseudobulk_{args.method}_{sampling_id}.csv"))
    cell_type_proportions.to_csv(os.path.join(file_dir, f"cell_type_proportions_pseudobulk_{'_'.join(args.cell_type_list)}_{args.method}_{sampling_id}.csv"))

    # calculate mean gene expression dataframe
    for cell_type in args.cell_type_list:
        # calculate for different cell types
        if cell_type == "HSC_MPP":
            cells_by_samples = (our_cdata.obs.loc[our_cdata.obs['cell_type'].isin(cell_type.split("_"))]
            .groupby('sample_id')
            .apply(lambda g: sorted(g.index))
            .to_dict())
        else:
            cells_by_samples = (our_cdata.obs.loc[our_cdata.obs['cell_type'].isin(normalize_cell_types(cell_type))]
            .groupby('sample_id')
            .apply(lambda g: sorted(g.index))
            .to_dict())
        
        # check if the ordering is correct
        assert(set(list(cells_by_samples.keys())) == set(cell_type_proportions.index.values))
        
        # calculate mean expression for each sample
        # sum raw counts in each sample, then normalize with library size and log1p
        
        # 1) Prepare flattened cell list and group codes (as you already did)
        pids = list(cells_by_samples.keys())
        all_cell_ids = []
        codes = []
        for j, pid in enumerate(pids):
            ids = list(cells_by_samples[pid])
            all_cell_ids.extend(ids)
            codes.extend([j] * len(ids))
        codes = np.asarray(codes, dtype=np.int64)
        
        # Map cell labels to integer row positions
        row_pos = our_cdata.obs.index.get_indexer(all_cell_ids)
        if (row_pos < 0).any():
            missing = [all_cell_ids[i] for i in np.where(row_pos < 0)[0][:10]]
            raise ValueError(f"Some cell IDs not found in obs.index, e.g. {missing[:10]}")
        
        # 2) Subset once
        ad_sub = our_cdata[row_pos, :].copy()
        
        # Helper: get a per-cell counts matrix C (n_cells_selected x n_genes)
        # Prefer a 'counts' layer if present; otherwise fall back to expm1 of X.
        def get_counts_matrix(adata_subset):
            # Use counts layer if available
            if hasattr(adata_subset, "layers") and "counts" in adata_subset.layers:
                C = adata_subset.layers["counts"]
            else:
                C = adata_subset.X
            if sparse.issparse(C):
                return C.tocsr()
            else:
                return sparse.csr_matrix(np.asarray(C), copy = False)
        
        C = get_counts_matrix(ad_sub)  # shape: (n_selected_cells, n_genes)
        n_groups = len(pids)
        n_genes = ad_sub.n_vars
        
        # 3) Build sparse indicator for groups and aggregate by sum (pseudobulk counts)
        # counts_by_group will be (n_groups x n_genes)
        if sparse.issparse(C):
            C = C.tocsr()
            G = sparse.csr_matrix(
                (np.ones(C.shape[0], dtype=np.float32), (np.arange(C.shape[0]), codes)),
                shape=(C.shape[0], n_groups)
            )
            counts_by_group = (G.T @ C).astype(np.float64)  # still sparse
            counts_by_group = counts_by_group.toarray()
        else:
            # Dense path
            G = np.zeros((n_groups, C.shape[0]), dtype=np.float64)
            G[np.arange(n_groups), codes] = 1.0  # not memory efficient if many cells
            counts_by_group = (G @ np.asarray(C, dtype=np.float64))
        
        # Number of cells per sample/group
        n_cells_per_group = np.bincount(codes, minlength=n_groups).reshape(-1, 1)
        
        # Mean per-cell counts on linear scale, guard against division by zero
        mean_counts = counts_by_group / np.clip(n_cells_per_group, 1, None)
        
        # Log of the mean per-cell counts (adds a +1 pseudocount before log)
        log_mean = np.log1p(mean_counts)
        
        # Build DataFrame
        log_mean_df = pd.DataFrame(
            log_mean,
            index=pd.Index(pids, name="sample_id"),
            columns=ad_sub.var_names
        )
        
        # 4) Optional library-size normalization and log1p
        lib_sizes = counts_by_group.sum(axis=1, keepdims=True)
        lib_sizes[lib_sizes == 0] = 1.0  # avoid division by zero
        
        # A) Simple log of pseudobulk counts (no library normalization)
        # log_counts = np.log1p(counts_by_group)
        
        # B) Log-CPM (recommended): scale each sample by its library size, then log1p
        cpm = counts_by_group * (1e6 / lib_sizes)
        log_cpm = np.log1p(cpm)
        
        # 5) Build DataFrames
        # log_counts_df = pd.DataFrame(
        #     log_counts,
        #     index=pd.Index(pids, name="sample_id"),
        #     columns=ad_sub.var_names
        # )
        
        log_cpm_df = pd.DataFrame(
            log_cpm,
            index=pd.Index(pids, name="sample_id"),
            columns=ad_sub.var_names
        )
        
        # print(f"Pseudobulk matrix shapes: counts={counts_by_group.shape}, log1p(counts)={log_counts_df.shape}, log1p(CPM)={log_cpm_df.shape}")
        print(f"Pseudobulk matrix shapes: counts={counts_by_group.shape}, log1p(CPM)={log_cpm_df.shape}")

        # cell_type_proportions.to_csv(os.path.join(file_dir, f"cell_type_proportions_pseudobulk_{args.method}_{sampling_id}.csv"))
        log_cpm_df.to_csv(os.path.join(file_dir, f"mean_gene_expression_pseudobulk_{cell_type}_{args.method}_{sampling_id}.csv"))
        

