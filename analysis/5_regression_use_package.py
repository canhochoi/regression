from regression.estimators import RidgeRRR, RidgeRRRCV
from regression.preprocessing import CompositionalILR, StandardScalerX, CenterY, TrainTestSplit
from regression.checkfitting import CheckFitting
from regression.estimators import RRRBinEvaluator
from regression.plotting import Plotting

import argparse
import os
import torch
import numpy as np
import pandas as pd
from skbio.stats.composition import closure, multiplicative_replacement, ilr, ilr_inv, clr

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

# Parse arguments
args = parser.parse_args()

# path_dir = "/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/min_cells/mean_gene_expression/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Read CSVs; first column is an ID, so use it as index (equivalent to R's [, 2:ncol])
# cell_type_1 = "HSC_MPP"
mean_gene_expr_df_1 = pd.read_csv(
    # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/mean_gene_expression_harmony_sample_id.csv",
    # args.path_dir + "mean_gene_expression_pseudobulk_" + args.cell_type_1 + "_min_cells.csv",
    os.path.join(args.path_dir, f"mean_gene_expression_pseudobulk_{args.cell_type_1}_{args.method}.csv"),
    index_col=0
)

# cell_type_2 = "B"
# cell_type_2 = "NKT"

mean_gene_expr_df_2 = pd.read_csv(
    # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/mean_gene_expression_harmony_sample_id.csv",
    # args.path_dir + "mean_gene_expression_pseudobulk_" + args.cell_type_2 + "_min_cells.csv",
    os.path.join(args.path_dir, f"mean_gene_expression_pseudobulk_{args.cell_type_2}_{args.method}.csv"),
    index_col=0
)

cell_type_proportions_df = pd.read_csv(
    # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/cell_type_proportions_harmony_sample_id.csv",
    # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/cell_type_proportions_harmony_sample_id_avg_pseudobulk.csv",
    # "/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/max_counts/parallel/cell_type_proportions_pseudobulk_max_counts.csv",
    os.path.join(args.path_dir, f"cell_type_proportions_pseudobulk_{args.method}.csv"),
    index_col=0
)

# meta =  pd.read_csv("/data1/soldatr/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/max_counts/parallel/metadata_samples_pseudobulk_max_counts.csv")
# meta = pd.read_csv("/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/min_cells/mean_gene_expression/metadata_samples_pseudobulk_min_cells.csv")
meta = pd.read_csv(os.path.join(args.path_dir, f"metadata_samples_pseudobulk_{args.method}.csv"))


# check that the indices (sample IDs) are the same and in the same order
assert(np.all(mean_gene_expr_df_1.index == mean_gene_expr_df_2.index))
assert(np.all(mean_gene_expr_df_1.index == cell_type_proportions_df.index))
assert(np.all(mean_gene_expr_df_1.index == meta["sample_id"]))

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

# cell_type_1
f"Doing regression on {args.cell_type_1}"

X_design_1 = np.concatenate([mean_gene_expr_df_1.values.astype(np.float64),
                             Z.to_numpy()], axis=1)

RRRB_1 = RRRBinEvaluator(device)
metrics_by_bin_1 = RRRB_1.evaluate(X_design = X_design_1,
                                       Y_ilr = Y_imp_ilr, 
                                       train_test = train_test,
                                       ilr_tx = ilr_tx,
                                       cell_type_proportions_df = cell_type_proportions_df)

# cell_type_2
print(f"Doing regression on {args.cell_type_2}")

X_design_2 = np.concatenate([mean_gene_expr_df_2.values.astype(np.float64),
                             Z.to_numpy()], axis=1)

RRRB_2 = RRRBinEvaluator(device)
metrics_by_bin_2 = RRRB_2.evaluate(X_design = X_design_2,
                                   Y_ilr = Y_imp_ilr, 
                                   train_test = train_test,
                                   ilr_tx = ilr_tx,
                                   cell_type_proportions_df = cell_type_proportions_df)

# cell_type_1 + cell_type_2
cell_type_1_2 = args.cell_type_1 + "_" + args.cell_type_2
print(f"Doing regression on {cell_type_1_2}")

X_design_1_2 = np.concatenate([mean_gene_expr_df_1.values.astype(np.float64),
                             mean_gene_expr_df_2.values.astype(np.float64),
                             Z.to_numpy()], axis=1)


RRRB_1_2 = RRRBinEvaluator(device)
metrics_by_bin_1_2 = RRRB_1_2.evaluate(X_design = X_design_1_2,
                                           Y_ilr = Y_imp_ilr, 
                                           train_test = train_test,
                                           ilr_tx = ilr_tx,
                                           cell_type_proportions_df = cell_type_proportions_df)

print(f"Plotting")

Plotting = Plotting()

# Assuming you have an instance `plotter = Plotting()`
metric_name_list = ["cors_comp", "cors_clr_comp", "Rsq_comp", "Rsq_clr_comp"]

metrics_dict = {args.cell_type_1: metrics_by_bin_1, 
                args.cell_type_2: metrics_by_bin_2, 
                cell_type_1_2: metrics_by_bin_1_2}


fig, axes = Plotting.scatter_design_matrices(metrics_dict = metrics_dict, metric_names = metric_name_list)

# for results in refined folder
# plot_dir = args.path_dir + "plots/"
# for testing MEBEMP-L
plot_dir = args.path_dir + args.cell_type_1
os.makedirs(plot_dir, exist_ok = True)
file_path = os.path.join(plot_dir, f"{cell_type_1_2 }.pdf")
fig.savefig(file_path, format = "pdf", bbox_inches = "tight")




