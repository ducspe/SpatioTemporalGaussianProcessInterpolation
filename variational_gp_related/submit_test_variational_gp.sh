#!/bin/bash
#SBATCH --job-name=variational_gp_test
#SBATCH --account=your_HPC_account
#SBATCH --partition=your_HPC_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/full_path_to_repository_folder/variational_gp_related/bash_gp_logs/%x_%j.out

# Submits a testing job for Variational GP models on various checkpoints.

mkdir -p /full_path_to_repository_folder/variational_gp_related/bash_gp_logs

echo "Start running the variational GP tests"
source ~/.bashrc

module load Python/3.12.3
source /full_path_to_repository_folder/.gp_venv/bin/activate

DATA_PATH="/full_path_to_repository_folder/dataset/continuous_signal_demo_dataset.nc"
CHECKPOINT_FOLDER="/full_path_to_repository_folder/variationalgp_checkpoints_dynamic_mask_512inducingpoints_256batchsize/"
RESULTS_FOLDER="/full_path_to_repository_folder/variationalgp_results_dynamic_mask_512inducingpoints_256batchsize"

CHECKPOINT_FILES=(${CHECKPOINT_FOLDER}*.pth)

echo "Testing with checkpoints: ${CHECKPOINT_FILES[@]}"
echo "Results will be saved in: ${RESULTS_FOLDER}"

python /full_path_to_repository_folder/variational_gp_related/test_variational_gp.py \
    --checkpoints "${CHECKPOINT_FILES[@]}" \
    --results_folder "${RESULTS_FOLDER}" \
    --num_frames_test 10 \
    --data_path "$DATA_PATH"

echo "Successfully ran all the variational GP tests"
