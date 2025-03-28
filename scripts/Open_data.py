#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:55:55 2025

@author: gamal
"""
import os 
import gpm # noqa
import pandas as pd
import glob
import xarray as xr

def concatenate_parquet_files(input_dir, recursive=True):
    """
    Concatenates all Parquet files in the input directory into a single Parquet file.

    Args:
        input_dir (str): Path to the directory containing Parquet files.
        recursive (bool): Whether to search for files in subdirectories (default: True).

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    # Find all Parquet files in the directory
    file_pattern = "**/*.parquet" if recursive else "*.parquet"
    parquet_files = glob.glob(os.path.join(input_dir, file_pattern), recursive=recursive)

    if not parquet_files:
        raise ValueError(f"No Parquet files found in {input_dir}")

    # Read and concatenate all files
    df_list = [pd.read_parquet(file) for file in parquet_files]
    concatenated_df = pd.concat(df_list, ignore_index=True)

    return concatenated_df

def find_zarr_file_for_patch(row, zarr_directory, filename_pattern="*.zarr"):
    """
    Finds the corresponding Zarr file for a given row in the concatenated DataFrame.

    Args:
        row (pd.Series): A row from the concatenated DataFrame representing a patch.
        zarr_directory (str): Base directory containing Zarr files.
        filename_pattern (str): Pattern to search for Zarr files (default: "*.zarr").

    Returns:
        tuple: (Path to matching Zarr file, Selected patch dataset) or (None, None) if not found.
    """
    granule_id = str(row["gpm_granule_id"])
    patch_id = row["patch_id"]
    
    # Extract year and month from the time column
    time = pd.to_datetime(row["time"])  # Ensure time is in datetime format
    year, month = time.year, time.month

    # Construct the expected path
    search_path = os.path.join(zarr_directory, f"{year:04d}/{month:02d}", filename_pattern)

    # Search for matching Zarr files in the specific directory
    zarr_files = glob.glob(search_path)

    for zarr_file in zarr_files:
        if granule_id in os.path.basename(zarr_file):  # Match granule ID in filename
            ds_stacked = xr.open_zarr(zarr_file)
            return zarr_file, ds_stacked.isel(patch=patch_id)

    print(f"⚠️ No matching Zarr file found for granule_id: {granule_id} in {year}/{month}")
    return None, None


parquet_dir ="/ltenas2/data/GPM_STORM_DB/parquet"
zarr_dir = "/ltenas2/data/GPM_STORM_DB/zarr"
concatenated_df = concatenate_parquet_files(parquet_dir)

patch_row = concatenated_df.iloc[43]
zarr_path, zarr_patch = find_zarr_file_for_patch(patch_row, zarr_dir)
zarr_patch['precipRateNearSurface'].gpm.plot_image()
zarr_patch['label_image'].gpm.plot_image()
