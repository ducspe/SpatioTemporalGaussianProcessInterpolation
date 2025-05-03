#!/usr/bin/env python
"""
Utility functions and classes for Variational GP training and testing.

Contents:
- MaskType (Enum) for fixed vs. dynamic masking.
- MyVariationalGPModel (ApproximateGP): A variational GP with separate temporal/spatial kernels.
- Data preparation and masking functions.
- Metric calculations and plotting helpers.
"""

import torch
import gpytorch
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class MaskType(Enum):
    FIXED_MASK_FOR_ALL_FRAMES = (1, "fixed")
    DIFFERENT_MASK_FOR_EACH_FRAME = (2, "dynamic")

    def __init__(self, num_value, short_name):
        self.num_value = num_value
        self.short_name = short_name

def select_inducing_points(train_x, M=200):
    """
    Randomly select M points from train_x as inducing points.
    If M >= number of training points, all are used.
    """
    N = train_x.size(0)
    if M >= N:
        return train_x.clone()
    rand_indices = torch.randperm(N)[:M]
    return train_x[rand_indices].clone()

class MyVariationalGPModel(gpytorch.models.ApproximateGP):
    """
    Variational GP model with separate temporal and spatial kernels.
    """

    def __init__(self, train_x, train_y, likelihood, time_kernel, space_kernel, num_inducing=200):
        """
        Args:
            train_x (torch.Tensor): [N, 3]
            train_y (torch.Tensor): [N]
            likelihood (gpytorch.likelihoods.Likelihood)
            time_kernel (str)
            space_kernel (str)
            num_inducing (int)
        """
        # Variational strategy
        inducing_points = select_inducing_points(train_x, M=num_inducing)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()

        # Temporal kernel
        if time_kernel == "rbf":
            print("Using RBF kernel for temporal dimension (variational).")
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0])
            )
        elif time_kernel == "matern":
            print("Using Matern kernel for temporal dimension (variational).")
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(active_dims=[0])
            )
        elif time_kernel == "periodic":
            print("Using Periodic kernel for temporal dimension (variational).")
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(active_dims=[0])
            )
        elif time_kernel == "softdisabled":
            print("Soft-disabling the temporal kernel (variational).")
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[0])
            )
            self.temporal_kernel.outputscale = 1e-5
            self.temporal_kernel.raw_outputscale.requires_grad = False
            self.temporal_kernel.base_kernel.lengthscale = 1e-5
            self.temporal_kernel.base_kernel.raw_lengthscale.requires_grad = False
        elif time_kernel == "disabled":
            print("Temporal kernel disabled (variational).")
            self.temporal_kernel = None
        elif time_kernel == "spectral":
            print("Using SpectralMixtureKernel for temporal dimension (variational).")
            self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=2, ard_num_dims=1, active_dims=[0]
                )
            )
        else:
            raise ValueError(f"Unknown time_kernel: {time_kernel}")

        # Spatial kernel
        if space_kernel == "rbf":
            print("Using RBF kernel for spatial dimension (variational).")
            self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=[1, 2])
            )
        elif space_kernel == "matern":
            print("Using Matern kernel for spatial dimension (variational).")
            self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(active_dims=[1, 2])
            )
        elif space_kernel == "periodic":
            print("Using Periodic kernel for spatial dimension (variational).")
            self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(active_dims=[1, 2])
            )
        elif space_kernel == "spectral":
            print("Using SpectralMixtureKernel for spatial dimension (variational).")
            self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=2, ard_num_dims=2, active_dims=[1, 2]
                )
            )
        else:
            raise ValueError(f"Unknown space_kernel: {space_kernel}")

        # Combine kernels
        if self.temporal_kernel is not None:
            self.covar_module = self.spatial_kernel + self.temporal_kernel
        else:
            self.covar_module = self.spatial_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def prepare_data(data, num_frames, mask_type, mask_probability, start_time_index=0):
    """
    Prepare masked data for variational GP training/validation.

    Args:
        data (torch.Tensor): [num_frames, H, W]
        num_frames (int)
        mask_type (MaskType)
        mask_probability (float)
        start_time_index (int)

    Returns:
        (data_x, data_y): Tensors [N, 3] and [N].
    """
    combined_x = []
    combined_y = []

    if mask_type == MaskType.FIXED_MASK_FOR_ALL_FRAMES:
        bernoulli_mask = torch.from_numpy(
            np.random.choice(2, size=data[0].shape, p=[mask_probability, 1 - mask_probability])
        )

    for frame_idx in range(num_frames):
        if mask_type == MaskType.DIFFERENT_MASK_FOR_EACH_FRAME:
            bernoulli_mask = torch.from_numpy(
                np.random.choice(2, size=data[0].shape, p=[mask_probability, 1 - mask_probability])
            )

        frame_data = data[frame_idx]
        unmasked_indices = torch.where(bernoulli_mask == 1)
        time_indices = torch.full(
            (unmasked_indices[0].shape[0], 1),
            start_time_index + frame_idx,
            dtype=torch.float32
        )
        spatial_indices = torch.vstack(unmasked_indices).T.float()

        x_coords = torch.cat((time_indices, spatial_indices), dim=1)
        y_coords = torch.flatten(frame_data[bernoulli_mask == 1])

        combined_x.append(x_coords)
        combined_y.append(y_coords)

    data_x = torch.cat(combined_x, dim=0)
    data_y = torch.cat(combined_y, dim=0)

    return data_x, data_y

def compute_metrics(ground_truth, prediction):
    """
    Compute MAE and RMSE.

    Args:
        ground_truth (np.ndarray): [T, H, W].
        prediction (np.ndarray): [T, H, W].

    Returns:
        dict: 'MAE' and 'RMSE'.
    """
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()
    mae = np.mean(np.abs(gt_flat - pred_flat))
    rmse = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
    return {'MAE': mae, 'RMSE': rmse}

def compute_uncertainty_correlation(ground_truth, prediction, predicted_variance):
    """
    Correlation between predicted variance and squared errors.

    Args:
        ground_truth (np.ndarray)
        prediction (np.ndarray)
        predicted_variance (np.ndarray)

    Returns:
        float correlation coefficient.
    """
    error_sq = (ground_truth - prediction) ** 2
    var_flat = predicted_variance.flatten()
    err_flat = error_sq.flatten()

    if np.all(var_flat == var_flat[0]) or np.all(err_flat == err_flat[0]):
        return 0.0

    return np.corrcoef(var_flat, err_flat)[0, 1]

def generate_meshgrid(num_frames, height=17, width=48, start_time_index=0):
    """
    Generate a 3D meshgrid for time, y, x.

    Returns:
        tuple of np.ndarray with shape [num_frames, height, width].
    """
    return np.meshgrid(
        np.linspace(start_time_index, start_time_index + num_frames - 1, num_frames),
        np.linspace(0, height - 1, height),
        np.linspace(0, width - 1, width),
        indexing='ij'
    )

def run_inference(model, likelihood, coords_torch, num_frames, height=17, width=48):
    """
    Perform inference under fast_pred_var.

    Returns:
        mean, var => np.ndarrays of shape [num_frames, height, width].
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = likelihood(model(coords_torch))
    mean = output.mean.reshape(num_frames, height, width).cpu().numpy()
    var = output.variance.reshape(num_frames, height, width).cpu().numpy()
    return mean, var

def plot_section(pdf, section_name, ground_truth, reconstructed_mean, reconstructed_var, start_frame_idx):
    """
    Plot frames (GT, Mean, Var, Error) in a 4-panel layout for a PDF.

    Args:
        pdf (PdfPages)
        section_name (str)
        ground_truth (np.ndarray): [T, H, W]
        reconstructed_mean (np.ndarray): [T, H, W]
        reconstructed_var (np.ndarray): [T, H, W]
        start_frame_idx (int)
    """
    num_frames = ground_truth.shape[0]
    error_map_all = np.abs(ground_truth - reconstructed_mean)

    vmin_gt, vmax_gt = ground_truth.min(), ground_truth.max()
    vmin_mean, vmax_mean = reconstructed_mean.min(), reconstructed_mean.max()
    vmin_var, vmax_var = reconstructed_var.min(), reconstructed_var.max()
    vmin_err, vmax_err = error_map_all.min(), error_map_all.max()

    for i in range(num_frames):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
        fig.suptitle(f"{section_name} - Frame {start_frame_idx + i}",
                     fontsize=20, fontweight='bold')

        # Ground Truth
        im0 = axes[0].imshow(
            ground_truth[i], cmap='viridis', vmin=vmin_gt, vmax=vmax_gt
        )
        axes[0].set_title(f"GT Frame {start_frame_idx + i}")
        fig.colorbar(im0, ax=axes[0], shrink=0.6)

        # Reconstructed Mean
        im1 = axes[1].imshow(
            reconstructed_mean[i], cmap='viridis', vmin=vmin_mean, vmax=vmax_mean
        )
        axes[1].set_title("Reconstructed Mean")
        fig.colorbar(im1, ax=axes[1], shrink=0.6)

        # Reconstructed Var
        im2 = axes[2].imshow(
            reconstructed_var[i], cmap='viridis', vmin=vmin_var, vmax=vmax_var
        )
        axes[2].set_title("Reconstructed Var")
        fig.colorbar(im2, ax=axes[2], shrink=0.6)

        # Error Map
        im3 = axes[3].imshow(
            error_map_all[i], cmap='viridis', vmin=vmin_err, vmax=vmax_err
        )
        axes[3].set_title("Error Map")
        fig.colorbar(im3, ax=axes[3], shrink=0.6)

        pdf.savefig(fig)
        plt.close(fig)
