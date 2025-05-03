#!/usr/bin/env python
"""
Test (infer with) a Variational GP model on a spatio-temporal dataset using GPyTorch.

Steps:
1) Load one or more checkpoint files.
2) Rebuild the model, load state, and run inference on train/val/test subsets.
3) Generate PDF plots of predictions, variance, and errors.
4) Compute and compare metrics (MAE, RMSE, variance-error correlation).
"""

import argparse
import os

import torch
import gpytorch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from my_variational_gp_utils import (
    MaskType,
    MyVariationalGPModel,
    prepare_data,
    compute_metrics,
    compute_uncertainty_correlation,
    generate_meshgrid,
    run_inference,
    plot_section
)

def main():
    parser = argparse.ArgumentParser(description='Spatio-temporal Variational GP Testing Script')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to one or more checkpoint files to compare.')
    parser.add_argument('--results_folder', type=str, default='results',
                        help='Folder to store PDF results and metrics plots.')
    parser.add_argument('--num_frames_test', type=int, default=10,
                        help='Number of frames used for testing.')
    parser.add_argument('--test_mask_type', type=str, choices=['fixed', 'dynamic'], default=None,
                        help='Override the stored mask type if desired at test time.')
    parser.add_argument('--test_prob_masking', type=float, default=None,
                        help='Override the stored masking probability if desired at test time.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset (NetCDF).')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_folder, exist_ok=True)

    metrics_results = {}

    print("\n===== Testing / Inference (Variational GP) =====")
    print(f"Results will be saved to: '{args.results_folder}'")
    print(f"Dataset path: {args.data_path}")

    for checkpoint_path in args.checkpoints:
        print(f"\n--- Processing checkpoint: {checkpoint_path} ---")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        train_params = checkpoint['train_params']

        time_kernel = train_params['time_kernel']
        space_kernel = train_params['space_kernel']
        stored_mask_type_str = train_params['mask_type']
        stored_prob_masking = train_params['prob_masking']
        num_frames_train = train_params['num_frames_train']
        num_frames_val = train_params['num_frames_val']
        num_inducing_points = train_params.get('num_inducing_points', 200)

        num_frames_test = args.num_frames_test
        test_mask_type_str = args.test_mask_type or stored_mask_type_str
        test_prob_masking = args.test_prob_masking or stored_prob_masking

        if test_mask_type_str == "fixed":
            test_mask_type = MaskType.FIXED_MASK_FOR_ALL_FRAMES
        else:
            test_mask_type = MaskType.DIFFERENT_MASK_FOR_EACH_FRAME

        print(f"From checkpoint: time_kernel={time_kernel}, space_kernel={space_kernel}, "
              f"stored_mask_type={stored_mask_type_str}, stored_prob={stored_prob_masking}, "
              f"trainFrames={num_frames_train}, valFrames={num_frames_val}, "
              f"testFrames={num_frames_test}, num_inducing={num_inducing_points}.")

        total_frames = num_frames_train + num_frames_val + num_frames_test
        ds = xr.open_dataset(args.data_path)
        all_data = torch.from_numpy(ds['var1'].to_numpy()[:total_frames])

        # Normalize by training's max
        train_data_gt = all_data[:num_frames_train]
        max_train_val = torch.max(train_data_gt)
        train_data_gt = train_data_gt / max_train_val
        val_data_gt = all_data[num_frames_train:num_frames_train+num_frames_val] / max_train_val
        test_data_gt = all_data[num_frames_train+num_frames_val:] / max_train_val

        if stored_mask_type_str == 'fixed':
            mask_type_stored = MaskType.FIXED_MASK_FOR_ALL_FRAMES
        else:
            mask_type_stored = MaskType.DIFFERENT_MASK_FOR_EACH_FRAME

        train_x, train_y = prepare_data(
            data=train_data_gt,
            num_frames=num_frames_train,
            mask_type=mask_type_stored,
            mask_probability=stored_prob_masking,
            start_time_index=0
        )
        val_x, val_y = prepare_data(
            data=val_data_gt,
            num_frames=num_frames_val,
            mask_type=mask_type_stored,
            mask_probability=stored_prob_masking,
            start_time_index=num_frames_train
        )
        test_x, test_y = prepare_data(
            data=test_data_gt,
            num_frames=num_frames_test,
            mask_type=test_mask_type,
            mask_probability=test_prob_masking,
            start_time_index=num_frames_train + num_frames_val
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = MyVariationalGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            time_kernel=time_kernel,
            space_kernel=space_kernel,
            num_inducing=num_inducing_points
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

        model.to(device)
        likelihood.to(device)
        model.eval()
        likelihood.eval()

        base_name = os.path.basename(checkpoint_path).replace('.pth', '')
        pdf_filename = os.path.join(args.results_folder, f'{base_name}.pdf')

        with PdfPages(pdf_filename) as pdf:
            # Train frames
            train_mesh = generate_meshgrid(
                num_frames=num_frames_train,
                height=17,
                width=48,
                start_time_index=0
            )
            train_coords = np.stack(train_mesh, axis=-1).reshape(-1, 3)
            train_coords_torch = torch.from_numpy(train_coords).float().to(device)
            train_mean, train_var = run_inference(
                model, likelihood, train_coords_torch,
                num_frames=num_frames_train, height=17, width=48
            )
            plot_section(pdf, "Train Frames", train_data_gt.numpy(),
                         train_mean, train_var, start_frame_idx=0)

            # Validation frames
            val_mesh = generate_meshgrid(
                num_frames=num_frames_val,
                height=17,
                width=48,
                start_time_index=num_frames_train
            )
            val_coords = np.stack(val_mesh, axis=-1).reshape(-1, 3)
            val_coords_torch = torch.from_numpy(val_coords).float().to(device)
            val_mean, val_var = run_inference(
                model, likelihood, val_coords_torch,
                num_frames=num_frames_val, height=17, width=48
            )
            plot_section(pdf, "Validation Frames", val_data_gt.numpy(),
                         val_mean, val_var, start_frame_idx=num_frames_train)

            # Test frames
            test_mesh = generate_meshgrid(
                num_frames=num_frames_test,
                height=17,
                width=48,
                start_time_index=num_frames_train + num_frames_val
            )
            test_coords = np.stack(test_mesh, axis=-1).reshape(-1, 3)
            test_coords_torch = torch.from_numpy(test_coords).float().to(device)
            test_mean, test_var = run_inference(
                model, likelihood, test_coords_torch,
                num_frames=num_frames_test, height=17, width=48
            )
            plot_section(pdf, "Test Frames", test_data_gt.numpy(),
                         test_mean, test_var,
                         start_frame_idx=num_frames_train + num_frames_val)

        print(f"PDF with train/val/test plots saved: {pdf_filename}")

        # Compute metrics on the test set
        test_data_np = test_data_gt.numpy()
        metrics = compute_metrics(test_data_np, test_mean)
        corr_var_err = compute_uncertainty_correlation(test_data_np, test_mean, test_var)
        avg_var = np.mean(test_var)

        print(f"  Test metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"Corr={corr_var_err:.4f}, AvgVar={avg_var:.6f}")

        kernel_label = (
            f"time={time_kernel}, space={space_kernel}, "
            f"mask={test_mask_type_str}, prob={stored_prob_masking}"
        )
        metrics_results[kernel_label] = [
            metrics["MAE"],
            metrics["RMSE"],
            corr_var_err,
            avg_var
        ]

    print("\nThe final metrics dictionary before plotting is: ")
    for k, v in metrics_results.items():
        print(k, "->", v)

    print("\n===== Generating bar plots for metrics across all checkpoints =====")
    kernel_labels = sorted(list(metrics_results.keys()))
    mae_values = [metrics_results[k][0] for k in kernel_labels]
    rmse_values = [metrics_results[k][1] for k in kernel_labels]
    corr_values = [metrics_results[k][2] for k in kernel_labels]
    avg_var_values = [metrics_results[k][3] for k in kernel_labels]

    suffix = args.results_folder.rstrip('/').split('/')[-1]

    # 1) MAE & RMSE
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(kernel_labels))
    width = 0.35
    ax1.bar(x - width/2, mae_values, width, label='MAE')
    ax1.bar(x + width/2, rmse_values, width, label='RMSE')
    ax1.set_ylabel("Error")
    ax1.set_title("MAE and RMSE across Kernels (Variational GP)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernel_labels, rotation=90, ha="right")
    ax1.legend()
    plt.tight_layout()
    out_fig_path_mae_rmse = os.path.join(args.results_folder, f"metric_{suffix}_mae_rmse.png")
    plt.savefig(out_fig_path_mae_rmse, dpi=200)
    plt.close(fig1)
    print(f"Bar-plot (MAE & RMSE) saved to: {out_fig_path_mae_rmse}")

    # 2) Variance-Error Correlation & Avg Variance
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.bar(x - width/2, corr_values, width, label='Var-Error Corr.')
    ax2.bar(x + width/2, avg_var_values, width, label='Avg Variance')
    ax2.set_title("Uncertainty Measures across Kernels (Variational GP)")
    ax2.set_ylabel("Value")
    ax2.set_xticks(x)
    ax2.set_xticklabels(kernel_labels, rotation=90, ha="right")
    ax2.legend()
    plt.tight_layout()
    out_fig_path_corr = os.path.join(args.results_folder, f"metric_{suffix}_uncertainty.png")
    plt.savefig(out_fig_path_corr, dpi=200)
    plt.close(fig2)
    print(f"Bar-plot (Variance-Error Correlation & Avg Var) saved to: {out_fig_path_corr}")

    print("\nDone comparing variational kernels! All checkpoints processed in one run.")


if __name__ == "__main__":
    main()
