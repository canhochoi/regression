#!/bin/bash
#SBATCH --job-name=filter_anndata
#SBATCH --partition=componc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#per-gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G             # use per-GPU memory, not --mem


#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --output=output_logs/filter_%a.out
#SBATCH --error=output_logs/filter_%a.err

echo "MEM_PER_NODE=${SLURM_MEM_PER_NODE:-} MEM_PER_CPU=${SLURM_MEM_PER_CPU:-} MEM_PER_GPU=${SLURM_MEM_PER_GPU:-}"

# Load environment
source ~/.bashrc
source /home/lel2/luan/projects/cell_tissue_phenotype/uv_env/.venv/bin/activate

# Load CUDA module (if required by your cluster)
# module load cuda/12.0  # Adjust version to match your system

# Verify GPU availability
#nvidia-smi || { echo "Error: No CUDA-capable device detected"; exit 1; }

# Define cell type sets and methods
#cell_type_sets=("Bcells" "Monocytes" "HSC MPP")
cell_type_sets=("HSC_MPP" "B" "BEMP" "CLP-E" "CLP-L" "CLP-M" "DC" "EP" "GMP-E" "GMP-L" "MEBEMP-E" "MEBEMP-L" "Monocytes" "NKT" "NKTDP")
methods=("min_cells" "max_counts")
#methods="min_cells"
#methods="max_counts"

# Calculate indices
num_methods=${#methods[@]}
cell_type_idx=$((SLURM_ARRAY_TASK_ID / num_methods))
method_idx=$((SLURM_ARRAY_TASK_ID % num_methods))

# Select cell types and method
cell_types=${cell_type_sets[$cell_type_idx]}
method=${methods[$method_idx]}

# Define paths
input_file="/data1/soldatr/luan/projects/cell_tissue_phenotype/results/our_cdata_raw_celltypes.h5ad"
#path_dir="../results/normal_HSC/pseudobulk/"
#path_dir="/data1/soldatr/luan/projects/cell_tissue_phenotype/running_slurm/test/"
#path_dir="/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/min_cells/mean_gene_expression/"
#path_dir="/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/max_counts/mean_gene_expression/"
path_dir="/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/refined/${method}/mean_gene_expression/"
python /home/lel2/luan/projects/cell_tissue_phenotype/scripts/4_EDA_celltype_gene_expr_pseudobulk_raw_counts.py --method "$method" --cell_type $cell_types --path_dir "$path_dir" 
