The scripts used to predict cell type proprotions from gene expression of stem cells (HSC_MPP) and terminal cells (e.g., B)
The procedure is as follows:

1. Obtain mean gene expression for each cell type 
```bash
sbatch script_SLURM_mean_gene_expr.sh
```
2. Do regression for stem cell and other terminal cells
```bash
sbatch script_SLURM_design_matrix.sh
```
Also check the notebook for results: 
```bash
5_regression_use_package.py
```
3. Check the cell numbers on regression quality by first getting sub-sampled mean gene expression for min(500, n_cells in each sample)
```bash
script_SLURM_sampling_manycelltypes.sh
```
4. Check notebook
```bash
5_regression_use_package-subsampling.ipynb
```

