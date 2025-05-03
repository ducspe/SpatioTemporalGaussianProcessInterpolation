#!/usr/bin/env python
"""
describe_dataset.py

This script loads a NetCDF dataset from the specified path and prints:
- The total number of frames (time dimension).
- The shape of the data array (T, Y, X).
- The minimum and maximum values in the dataset.

Usage example:
python describe_dataset.py --data_path /path/to/dataset.nc

Concrete example using the demo dataset file:
python additional_material/describe_dataset.py --data_path /full_path_to_repository_folder/dataset/continuous_signal_demo_dataset.nc
"""

import argparse
import xarray as xr
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Describe a spatio-temporal NetCDF dataset.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the NetCDF dataset (containing var1).')
    args = parser.parse_args()

    # Open the dataset
    ds = xr.open_dataset(args.data_path)

    # Extract the data variable named 'var1'
    data_var = ds['var1']

    # Convert to NumPy for quick inspection
    data_np = data_var.to_numpy()

    # Print basic info
    print("===== Dataset Description =====")
    print(f"File path: {args.data_path}")
    print("Variable name: var1")
    print(f"Shape (T, Y, X): {data_np.shape}")
    print(f"Number of frames (time dimension): {data_np.shape[0]}")
    print(f"Minimum value: {np.min(data_np)}")
    print(f"Maximum value: {np.max(data_np)}")
    print("================================")

if __name__ == "__main__":
    main()
