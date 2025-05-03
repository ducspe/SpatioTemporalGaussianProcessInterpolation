#!/usr/bin/env python
"""
Train an Exact GP model on a spatio-temporal dataset using GPyTorch.

Steps:
1) Parse command-line arguments (kernel choices, frames, dataset path, etc.).
2) Load and normalize the dataset from a NetCDF file.
3) Prepare training and validation sets (applying masks to simulate missing data).
4) Define an Exact GP model with specified temporal and spatial kernels.
5) Train the model, tracking losses, and save the best checkpoint.
"""

import argparse
import os

import torch
import gpytorch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from my_exact_gp_utils import (
    MaskType,
    MyExactGPModel,
    prepare_data
)

def main():
    parser = argparse.ArgumentParser(description='Spatio-temporal Exact GP Training Script')
    parser.add_argument('--prob_masking', type=float, default=0.5,
                        help='Probability of masking data points.')
    parser.add_argument('--max_iters', type=int, default=2,
                        help='Maximum training iterations (epochs).')
    parser.add_argument('--num_frames_train', type=int, default=50,
                        help='Number of frames to use for training.')
    parser.add_argument('--num_frames_valid', type=int, default=10,
                        help='Number of frames to use for validation.')
    parser.add_argument('--mask_type', type=str, choices=['fixed', 'dynamic'], default='fixed',
                        help='Mask type (fixed mask or dynamic mask across frames).')
    parser.add_argument('--timekernel', type=str,
                        choices=['rbf', 'matern', 'periodic', 'softdisabled', 'disabled', 'spectral'],
                        default='rbf',
                        help='Kernel choice for the temporal dimension.')
    parser.add_argument('--spacekernel', type=str,
                        choices=['rbf', 'matern', 'periodic', 'spectral'],
                        default='rbf',
                        help='Kernel choice for the spatial dimension.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset (NetCDF).')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_FRAMES_TRAIN = args.num_frames_train
    NUM_FRAMES_VALID = args.num_frames_valid
    PROB_MASKING = args.prob_masking
    MAX_ITERS = args.max_iters

    # Determine the mask type
    if args.mask_type == "fixed":
        USED_MASK_TYPE = MaskType.FIXED_MASK_FOR_ALL_FRAMES
    else:
        USED_MASK_TYPE = MaskType.DIFFERENT_MASK_FOR_EACH_FRAME

    print("\n===== Exact GP Training =====")
    print(f"Device: {device}")
    print(f"Dataset path: {args.data_path}")
    print(f"Probability of masking: {PROB_MASKING}")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Train frames: {NUM_FRAMES_TRAIN}, Validation frames: {NUM_FRAMES_VALID}")
    print(f"Mask type: {USED_MASK_TYPE.short_name}")
    print(f"Time kernel: {args.timekernel}, Space kernel: {args.spacekernel}")

    if device == 'cuda':
        print("Clearing CUDA cache before starting the experiment.")
        torch.cuda.empty_cache()

    # Load and normalize data
    ds = xr.open_dataset(args.data_path)
    all_data = torch.from_numpy(ds['var1'].to_numpy())
    print("all_data.shape:", all_data.shape)

    train_data = all_data[:NUM_FRAMES_TRAIN]
    max_train_val = torch.max(train_data)
    train_data = train_data / max_train_val

    val_data = all_data[NUM_FRAMES_TRAIN:NUM_FRAMES_TRAIN + NUM_FRAMES_VALID] / max_train_val
    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")

    # Prepare training/validation sets
    train_data_x, train_data_y = prepare_data(
        data=train_data,
        num_frames=NUM_FRAMES_TRAIN,
        mask_type=USED_MASK_TYPE,
        mask_probability=PROB_MASKING,
        start_time_index=0
    )
    val_data_x, val_data_y = prepare_data(
        data=val_data,
        num_frames=NUM_FRAMES_VALID,
        mask_type=USED_MASK_TYPE,
        mask_probability=PROB_MASKING,
        start_time_index=NUM_FRAMES_TRAIN
    )

    print(f"Train X shape: {train_data_x.shape}, Y shape: {train_data_y.shape}")
    print(f"Val   X shape: {val_data_x.shape},   Y shape: {val_data_y.shape}")

    # Build the Exact GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MyExactGPModel(
        train_x=train_data_x,
        train_y=train_data_y,
        likelihood=likelihood,
        time_kernel=args.timekernel,
        space_kernel=args.spacekernel
    )

    model.to(device)
    likelihood.to(device)
    train_data_x = train_data_x.to(device)
    train_data_y = train_data_y.to(device)
    val_data_x = val_data_x.to(device)
    val_data_y = val_data_y.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_filename = (
        f"bestmodel__time{args.timekernel}__space{args.spacekernel}"
        f"__mask{USED_MASK_TYPE.short_name}__prob{PROB_MASKING}"
        f"__train{NUM_FRAMES_TRAIN}__val{NUM_FRAMES_VALID}.pth"
    )
    best_model_path = os.path.join(args.checkpoint_dir, best_model_filename)

    best_val_loss = float('inf')
    best_iter = 0

    # Training loop
    for iter_idx in range(MAX_ITERS):
        optimizer.zero_grad()
        output = model(train_data_x)
        loss = -mll(output, train_data_y)  # Negative log marginal likelihood
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            val_output = model(val_data_x)
            val_loss = -mll(val_output, val_data_y)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_iter = iter_idx + 1
            checkpoint_dict = {
                'epoch': iter_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'loss': loss.item(),
                'train_params': {
                    'time_kernel': args.timekernel,
                    'space_kernel': args.spacekernel,
                    'mask_type': USED_MASK_TYPE.short_name,
                    'prob_masking': PROB_MASKING,
                    'num_frames_train': NUM_FRAMES_TRAIN,
                    'num_frames_val': NUM_FRAMES_VALID
                }
            }
            torch.save(checkpoint_dict, best_model_path)
            print(f"New best model saved at iteration {best_iter} (val loss: {best_val_loss:.4f})")

        print(f"Iter {iter_idx+1}/{MAX_ITERS} - "
              f"Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}")

        model.train()
        likelihood.train()

    print(f"\nTraining complete. Best iteration: {best_iter}, best val loss: {best_val_loss:.3f}")
    print("===== End of Training =====")


if __name__ == "__main__":
    main()
