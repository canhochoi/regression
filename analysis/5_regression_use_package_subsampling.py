from regression.estimators import RidgeRRR, RidgeRRRCV
from regression.preprocessing import CompositionalILR, StandardScalerX, CenterY, TrainTestSplit
from regression.checkfitting import CheckFitting
from regression.estimators import RRRBinEvaluator
from regression.plotting import Plotting
from regression.sampling import SamplingMetricsSummarizer

import argparse
import os
import torch
import numpy as np
import pandas as pd
from skbio.stats.composition import closure, multiplicative_replacement, ilr, ilr_inv, clr
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    choices=["max_counts", "min_cells"],
    default="min_cells",
    help="Filtering method to use: 'max_counts' or 'min_cells' (default: min_cells)"
    )
parser.add_argument(
    "--cell_type_1",
    type=str,
    # nargs="+",  #remove to get plain string
    required=True,
    help="Celltypes to calculate, space-separated (e.g., HSC_MPP)"
    )
parser.add_argument(
    "--cell_type_2",
    type=str,
    # nargs="+",
    required=True,
    help="Celltypes to calculate, space-separated (e.g., B)"
    )
parser.add_argument(
    "--path_dir",
    type=str,
    help="Folder to store results (e.g., ../results/normal_HSC/pseudobulk/)"
    )
parser.add_argument(
    "--n_sampling",
    type=int,
    help="Number of samplings"
    )
parser.add_argument(
    "--path_dir_org",
    type=str,
    help="Folder to store original cell type proportions and meta data (e.g., ../results/normal_HSC/pseudobulk/)"
    )

# Parse arguments
args = parser.parse_args()

# path_dir = "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/min_cells/mean_gene_expression/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric_name_list = ["cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"]

cell_type_1 = args.cell_type_1
cell_type_2 = args.cell_type_2
cell_type_3 = cell_type_1 + "_" + cell_type_2
path_dir = args.path_dir
n_sampling = int(args.n_sampling)
method = args.method
org_dir = args.path_dir_org

Sampling_Summarizer = SamplingMetricsSummarizer()

summary_df_dict = {}  # initialize dictionary

for sampling_id in range(n_sampling):
    print(f"Doing {sampling_id}")
    # 1) Read CSVs; first column is an ID, so use it as index (equivalent to R's [, 2:ncol])
    
    
    mean_gene_expr_df_HSC_MPP = pd.read_csv(
        # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/mean_gene_expression_harmony_sample_id.csv",
        os.path.join(path_dir, f"{cell_type_3}/mean_gene_expression_pseudobulk_{cell_type_1}_{method}_{sampling_id}.csv"),
        index_col=0
    )
    
    # ncells_per_sample_df_HSC_MPP = pd.read_csv(path_dir_2 + cell_type_1 + "_max_counts.csv", index_col = 0)
    
        
    # cell_type_2 = "NKT"
    
    mean_gene_expr_df_B = pd.read_csv(
        # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/mean_gene_expression_harmony_sample_id.csv",
        os.path.join(path_dir, f"{cell_type_3}/mean_gene_expression_pseudobulk_{cell_type_2}_{method}_{sampling_id}.csv"),
        index_col=0
    )
    
    # ncells_per_sample_df_B = pd.read_csv(path_dir_2 + cell_type_2 + "_max_counts.csv", index_col = 0)
    
    cell_type_proportions_df = pd.read_csv(
        # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/cell_type_proportions_harmony_sample_id.csv",
        # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/cell_type_proportions_harmony_sample_id_avg_pseudobulk.csv",
        # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/max_counts/parallel/cell_type_proportions_pseudobulk_max_counts.csv",
        # "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/weighted_ncells/cell_type_proportions_pseudobulk_max_counts.csv",
                os.path.join(org_dir, f"{method}/mean_gene_expression/cell_type_proportions_pseudobulk_{method}.csv"),
        index_col=0
    )
    
    # meta =  pd.read_csv("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/max_counts/parallel/metadata_samples_pseudobulk_max_counts.csv")
    # meta = pd.read_csv("/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/weighted_ncells/metadata_samples_pseudobulk_max_counts.csv")
    meta = pd.read_csv(os.path.join(org_dir, f"{method}/mean_gene_expression/metadata_samples_pseudobulk_{method}.csv"))
    
    
    # check that the indices (sample IDs) are the same and in the same order
    assert(np.all(mean_gene_expr_df_HSC_MPP.index == mean_gene_expr_df_B.index))
    assert(np.all(mean_gene_expr_df_HSC_MPP.index == cell_type_proportions_df.index))
    assert(np.all(mean_gene_expr_df_HSC_MPP.index == meta["sample_id"]))
    # assert(np.all(ncells_per_sample_df_B.index == meta["sample_id"]))
    # assert(np.all(ncells_per_sample_df_HSC_MPP.index == meta["sample_id"]))
    
    # ilr-transform cell type proportions
    ilr_tx = CompositionalILR(zero_replacement=True).fit(cell_type_proportions_df.values)
    Y_imp_ilr = ilr_tx.transform(cell_type_proportions_df.values)  # replace your earlier line
    
    # Covariates: sex (one-hot, drop_first) and ages
    Z = pd.get_dummies(meta[['sex_src']], prefix='sex_src', drop_first=True)
    Z['ages'] = meta['ages'].astype(float)
    Z = Z.astype(float)
    Z.index = meta.index  # align
    
    
    # stem cells of HSC_MPP and B cells
    # X_design = np.concatenate([mean_gene_expr_df_HSC_MPP.values.astype(np.float64),
    #                            mean_gene_expr_df_B.values.astype(np.float64),
    #                            Z.to_numpy()], axis=1)
    
    # generate bins
    train_test = TrainTestSplit(nfolds = 5, random_state = 0)

    # stem cells of HSC_MPP
    X_design_HSC_MPP = np.concatenate([mean_gene_expr_df_HSC_MPP.values.astype(np.float64),
                               Z.to_numpy()], axis=1)
    
    RRRB_HSC = RRRBinEvaluator(device)
    # do for nfolds
    # sample_weight = np.array(ncells_per_sample_df_HSC_MPP['n_cells'].tolist())
    # sample_weight = np.log10(np.array(ncells_per_sample_df_HSC_MPP['n_cells'].tolist()))
    
    metrics_by_bin_HSC = RRRB_HSC.evaluate(X_design = X_design_HSC_MPP,
                                           Y_ilr = Y_imp_ilr, 
                                           train_test = train_test,
                                           ilr_tx = ilr_tx,
                                           cell_type_proportions_df = cell_type_proportions_df,
                                           verbose = False)
    # concatenate cell types
    
    X_design_HSC_MPP_B = np.concatenate([mean_gene_expr_df_HSC_MPP.values.astype(np.float64),
                                         mean_gene_expr_df_B.values.astype(np.float64),
                                         Z.to_numpy()], axis=1)
    
    
    RRRB_HSC_B = RRRBinEvaluator(device)
    
    # ncells_HSC_MPP_B = ncells_per_sample_df_HSC_MPP['n_cells'] + ncells_per_sample_df_B['n_cells']
    # sample_weight = np.asarray(ncells_HSC_MPP_B.tolist())
    
    # sample_weight = np.log10(np.asarray(ncells_HSC_MPP_B.tolist()))
    
    metrics_by_bin_HSC_MPP_B = RRRB_HSC_B.evaluate(X_design = X_design_HSC_MPP_B,
                                               Y_ilr = Y_imp_ilr, 
                                               train_test = train_test,
                                               ilr_tx = ilr_tx,
                                               cell_type_proportions_df = cell_type_proportions_df,
                                               verbose = False)
    
    
    # another cell type

    X_design_B = np.concatenate([mean_gene_expr_df_B.values.astype(np.float64),
                           Z.to_numpy()], axis=1)

    RRRB_B = RRRBinEvaluator(device)
    
    # sample_weight = np.array(ncells_per_sample_df_B['n_cells'].tolist())
    
    # sample_weight = np.log10(np.array(ncells_per_sample_df_B['n_cells'].tolist()))
    
    
    metrics_by_bin_B = RRRB_B.evaluate(X_design = X_design_B,
                                       Y_ilr = Y_imp_ilr, 
                                       train_test = train_test,
                                       ilr_tx = ilr_tx,
                                       cell_type_proportions_df = cell_type_proportions_df,
                                       verbose = False)
    

    # Plotting = Plotting()
    
    # Assuming you have an instance `plotter = Plotting()`
    # metric_name_list = ["cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"]
    
    metrics_dict = {cell_type_1: metrics_by_bin_HSC, 
                    cell_type_2: metrics_by_bin_B, 
                    cell_type_3: metrics_by_bin_HSC_MPP_B}
    
    # Summarize one sampling run
    summary_df = Sampling_Summarizer.summarize_metrics(metrics_dict, metric_names=metric_name_list, run_id=sampling_id)
    
    summary_df_dict[sampling_id] = summary_df

#Summarize all samplings
df_across = Sampling_Summarizer.summarize_across_samplings(summary_df_dict)

print(f"Plotting")

Plotting = Plotting()

# Assuming you have an instance `plotter = Plotting()`
metric_name_list = ["cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"]

fig, axes = Plotting.plot_sampling_summary_scatter_panel(df_across, 
                                                         metric_name_list)

plot_dir = os.path.join(path_dir, f"plots/")
os.makedirs(plot_dir, exist_ok = True)
file_path = os.path.join(plot_dir, f"{cell_type_3}.pdf")
fig.savefig(file_path, format = "pdf", bbox_inches = "tight")

# save the dataframe
import pickle
file_path = os.path.join(plot_dir, f"summary_df_dict_{cell_type_3}.pkl")
with open(file_path, "wb") as f:
    pickle.dump(summary_df_dict, f)
    
# to load back
# with open(file_path, "rb") as f:
#     summary_df_dict = pickle.load(f)
    


