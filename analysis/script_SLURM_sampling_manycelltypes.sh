#!/usr/bin/env bash
#SBATCH --job-name=ct_pseudobulk_pairs
#SBATCH --partition=componc_gpu          # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # one Python process at a time
#SBATCH --cpus-per-task=8                # adjust as needed
#SBATCH --mem=128G                       # sufficient RAM for large AnnData ops
#SBATCH --gres=gpu:1                     # one GPU for the job
#SBATCH --time=12:00:00                  # walltime; increase if needed
#SBATCH --output=output_logs/%x_%j.out   # stdout log
#SBATCH --error=output_logs/%x_%j.err    # stderr log

#set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
# Ensure logs dir exists before submission (or create it in a wrapper script)
# mkdir -p output_logs

# Activate your Python environment
source ~/.bashrc
source ../uv_env/.venv/bin/activate
# If using conda instead, comment the line above and use:
# source /data1/soldatr/luan/Python/miniforge3/etc/profile.d/conda.sh
# conda activate FigRemake

# Diagnostics: node and GPU info
echo "Node: $(hostname)"
echo "Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID})"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi not available"

# -----------------------------
# Inputs and parameters
# -----------------------------

# Anchor cell type to pair with others
anchor="HSC_MPP"

# All cell types (include anchor so we can filter it out)
cell_type_sets=(
  "HSC_MPP" "B" "BEMP" "CLP-E" "CLP-L" "CLP-M" "DC" "EP"
  "GMP-E" "GMP-L" "MEBEMP-E" "MEBEMP-L" "Monocytes" "NKT" "NKTDP"
)
cell_type_sets=("Monocytes" "NKT" "NKTDP")
# Filtering methods to run
methods=("max_counts")  # or ("min_cells" "max_counts")

# Output directory (created if missing)
path_dir="/home/lel2/luan/projects/cell_tissue_phenotype/results/normal_HSC/pseudobulk/subsampling/"
mkdir -p "${path_dir}"

# Sampling controls
n_sampling=100
ncells_sampling=500  # pass if your script uses this argparse option

# -----------------------------
# Sequential processing
# -----------------------------
for method in "${methods[@]}"; do
  echo "=== Running method: ${method} ==="
  for ct in "${cell_type_sets[@]}"; do
    # Skip pairing the anchor with itself
    if [[ "$ct" == "$anchor" ]]; then
      continue
    fi

    echo ">>> Pair: ${anchor} + ${ct} (method=${method}, n_sampling=${n_sampling}, ncells_sampling=${ncells_sampling})"

    # Use srun with the job's allocation (matching your working script)
#    srun -N1 -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gres=gpu:1 \
     python ../scripts/4_EDA_celltype_gene_expr_pseudobulk_raw_counts_subsampling_many_celltypes.py \
        --method "${method}" \
        --cell_type_list "${anchor}" "${ct}" \
        --path_dir "${path_dir}" \
        --n_sampling "${n_sampling}" \
        --ncells_sampling "${ncells_sampling}"

    # Optional short pause
    sleep 2
  done
done

echo "All done."
