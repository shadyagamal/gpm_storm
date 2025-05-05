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
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from gpm.bucket.writers import preprocess_writer_kwargs

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

def find_zarr_file_for_patchs(row, zarr_directory, filename_pattern="*.zarr"):
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
    
    # If it does not find the file it is in the folder of the month before 
    prev_time = time - pd.DateOffset(months=1)
    year, month = prev_time.year, prev_time.month
    for zarr_file in zarr_files:
        if granule_id in os.path.basename(zarr_file):  # Match granule ID in filename
            ds_stacked = xr.open_zarr(zarr_file)
            return zarr_file, ds_stacked.isel(patch=patch_id)
    print(f"⚠️ No matching Zarr file found for granule_id: {granule_id} in {year}/{month}")
    return None, None

def find_zarr_file_for_patch(row, zarr_directory, filename_pattern="*.zarr"):
    granule_id = str(row["gpm_granule_id"])
    patch_id = row["patch_id"]
    
    time = pd.to_datetime(row["time"])
    year, month = time.year, time.month

    def try_find_file(y, m):
        search_path = os.path.join(zarr_directory, f"{y:04d}/{m:02d}", filename_pattern)
        zarr_files = glob.glob(search_path)
        for zarr_file in zarr_files:
            if granule_id in os.path.basename(zarr_file):
                ds_stacked = xr.open_zarr(zarr_file)
                if patch_id < ds_stacked.sizes["patch"]:
                    return zarr_file, ds_stacked.isel(patch=patch_id)
        return None, None

    # Try current month
    result = try_find_file(year, month)
    if result[0] is not None:
        return result

    # Try previous month
    prev_time = time - pd.DateOffset(months=1)
    prev_year, prev_month = prev_time.year, prev_time.month
    result = try_find_file(prev_year, prev_month)
    if result[0] is not None:
        return result

    print(f"No matching Zarr file found for granule_id: {granule_id} in {year}/{month} or {prev_time.year}/{prev_time.month:02d}")
    return None, None

def concatenate_parquet_files_arrow(input_dir, output_dir, row_group_size="200MB", max_file_size="2GB"):
    """
    Efficiently concatenates all Parquet files in a directory using PyArrow.

    Args:
        input_dir (str): Path to the directory containing Parquet files.
        output_dir (str): Destination directory for the merged dataset.
        row_group_size (str): Row group size for efficient writing.
        max_file_size (str): Maximum file size per output Parquet file.
    
    Returns:
        None
    """
    # Find all Parquet files
    parquet_files = glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True)
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {input_dir}")

    # Read dataset using PyArrow
    dataset = ds.dataset(parquet_files, format="parquet")

    # Load a template table from one of the Parquet files
    template_table = pq.read_table(parquet_files[0])

    # Define writer options
    writer_kwargs = {
        "row_group_size": row_group_size,
        "max_file_size": max_file_size,
        "compression": "snappy",
        "compression_level": None,
        "max_open_files": 0,
        "use_threads": True,
        "write_metadata": False,
        "write_statistics": False,
    }

    # Process writer kwargs with a template table
    writer_kwargs, metadata_collector = preprocess_writer_kwargs(
        writer_kwargs=writer_kwargs, df=template_table
    )

    # Define scanner for reading data efficiently
    scanner = dataset.scanner(
        batch_size=131_072,
        batch_readahead=10,
        fragment_readahead=20,
        use_threads=True,
    )

    # Write concatenated dataset to the output directory
    ds.write_dataset(
        scanner,
        base_dir=output_dir,
        basename_template="merged_data_{i}.parquet",
        create_dir=True,
        existing_data_behavior="overwrite_or_ignore",
        **writer_kwargs,
    )

    print(f"Concatenated Parquet files saved to {output_dir}")

parquet_dir ="/ltenas2/data/GPM_STORM_DB/parquet"
output_dir = "/ltenas2/data/GPM_STORM_DB/merged"
zarr_dir = "/ltenas2/data/GPM_STORM_DB/zarr"

concatenate_parquet_files_arrow(parquet_dir, output_dir)
#concatenated_df = concatenate_parquet_files(parquet_dir)

df = pd.read_parquet("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet")
print(df.head())
print(df.info())
df1 = pd.read_parquet("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet")


# for i in range(len(df)):
#     print(i)
#     patch_row = df.iloc[i]
#     zarr_path, zarr_patch = find_zarr_file_for_patch(patch_row, zarr_dir)
    
# bad_indices = []
# #last checked 171665
# bad_indicess = []

# for i in range(24696,len(df)):
#     patch_row = df.iloc[i]
#     try:
#         zarr_path, ds = find_zarr_file_for_patch(patch_row, zarr_dir)
#         if ds is None:
#             bad_indicess.append(i)
#     except IndexError as e:
#         print(f"IndexError at row {i}: {e}")
#         bad_indicess.append(i)
#     except Exception as e:
#         print(f"Unexpected error at row {i}: {e}")
#         bad_indicess.append(i)
    
patch_row = df.iloc[8297]
zarr_path, zarr_patch = find_zarr_file_for_patch(patch_row, zarr_dir)
zarr_patch['precipRateNearSurface'].gpm.plot_image()
zarr_patch['label_image'].gpm.plot_image()