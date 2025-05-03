#!/bin/bash
#SBATCH --job-name=exact_gp_train
#SBATCH --account=your_HPC_account
#SBATCH --partition=your_HPC_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/full_path_to_repository_folder/exact_gp_related/bash_gp_logs/%x_%j.out

# Submits multiple Exact GP training jobs with different kernel and mask settings on an HPC system.

mkdir -p /full_path_to_repository_folder/exact_gp_related/bash_gp_logs

start_time=$(date +%s)
echo "Start running the experiments"
source ~/.bashrc

module load Python/3.12.3
source /full_path_to_repository_folder/.gp_venv/bin/activate

DATA_PATH="/full_path_to_repository_folder/dataset/continuous_signal_demo_dataset.nc"

# 1) SpectralMixture kernels for both time and space
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel spectral --spacekernel spectral \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel spectral --spacekernel spectral \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel spectral --spacekernel spectral --num_frames_train 30 \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

# 2) Disabled time kernel, with RBF in space
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel disabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel disabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel disabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

# 3) Soft-disabling the temporal kernel, RBF in space
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel softdisabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel softdisabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel softdisabled --spacekernel rbf \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

# 4) RBF kernels for both time and space
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel rbf --spacekernel rbf \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel rbf --spacekernel rbf \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel rbf --spacekernel rbf \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

# 5) Matern kernels
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel matern --spacekernel matern \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel matern --spacekernel matern \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel matern --spacekernel matern \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

# 6) Periodic kernels
python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel periodic --spacekernel periodic \
    --max_iters 100 --prob_masking 0.9 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel periodic --spacekernel periodic \
    --max_iters 100 --prob_masking 0.8 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

python /full_path_to_repository_folder/exact_gp_related/train_exact_gp.py \
    --data_path "$DATA_PATH" \
    --timekernel periodic --spacekernel periodic --num_frames_train 30 \
    --max_iters 100 --prob_masking 0.5 --mask_type dynamic \
    --checkpoint_dir "exactgp_checkpoints_dynamic_mask"

end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(bc <<< "scale=2; ${runtime} / 60")
echo "Total runtime for all experiments: ${runtime_minutes} minutes"
