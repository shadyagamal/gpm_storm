#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download GPM data, label storms, extract patches, and compute statistics.

@author: shadya
"""

# IMPORTS
import sys
import os
import datetime
import gpm # type: ignore
import numpy as np 
import pandas as pd 
import ximage # noqa


BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
PACKAGE_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm_storm.gpm_storm.features.image import calculate_image_statistics


def download_gpm_data(start_time, end_time, product="2A-DPR", product_type="RS", version=7):
    """
    Download GPM data for a given time period.
    """
    print("Downloading GPM data...")
    gpm.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        force_download=False,
        verbose=True,
        progress_bar=True,
        check_integrity=False,
    )
    print("Download complete!\n")
    return None
 
#If working with all of the locally downloaded files working with the granule 
""" filepath_list = get_local_filepaths(product, version=version, product_type=product_type)
filepath_2023 = [fp for fp in filepath_list if "/2023/" in fp]

for filepath in filepath_2023:
    ds = gpm.open_granule(filepath, variables=variables, scan_mode="FS") """

def load_gpm_data(start_time, end_time, product="2A-DPR", product_type="RS", version=7):
    """
    Load GPM dataset for a given time period.
    """
    print("Loading GPM dataset...")
    try:
        ds = gpm.open_dataset(
            product=product,
            product_type=product_type,
            version=version,
            start_time=start_time,
            end_time=end_time,
            variables=VARIABLES,
            prefix_group=False,
        )
        print("Dataset loaded successfully!\n")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    return None


def label_storms(ds):
    """
    Label precipitation patches in the dataset.
    """
    print("Labeling storms...")
    da = ds["precipRateNearSurface"].compute()

    xr_obj = da.ximage.label(
        min_value_threshold=0.05,
        max_value_threshold=np.inf,
        min_area_threshold=5,
        max_area_threshold=np.inf,
        footprint=5,
        sort_by="area",
        sort_decreasing=True,
        label_name="label",
    )
    print("Storm labeling complete!\n")
    return xr_obj


def extract_patches(xr_obj):
    """
    Extract patches from labeled storm data.
    """
    print("Extracting patches...")
    label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
        label_name="label",
        patch_size=(49, 20),
        variable="precipRateNearSurface",
        n_patches=500,
        n_labels=None,
        labels_id=None,
        padding=0,
        centered_on="label_bbox",
        partitioning_method=None,
        debug=False,
    )
    print(f"Extracted {len(label_isel_dict)} patches!\n")
    return label_isel_dict


def compute_patch_statistics(ds, label_isel_dict, n_patches=500):
    """
    Compute statistics for extracted patches.
    """
    print("Computing patch statistics...")

    ds["zFactorFinal"] = ds["zFactorFinal"].compute()
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    ds["sunLocalTime"] = ds["sunLocalTime"].compute()

    patch_statistics = [
        calculate_image_statistics(ds, label_isel_dict[i][0]) for i in range(1, n_patches)
    ]

    print("Patch statistics computed!\n")
    return patch_statistics


def save_results(patch_statistics, output_path):
    """
    Save computed statistics to a Parquet file.
    """
    print(f"Saving results to {output_path}...")
    df = pd.DataFrame(patch_statistics)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S.%f")

    df.to_parquet(output_path)
    print(f"Features saved successfully to {output_path}\n")
    return None

def main():
    # Define the time period for data download
    start_time = datetime.datetime.strptime("2023-08-20 20:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.datetime.strptime("2023-08-28 00:00:00", "%Y-%m-%d %H:%M:%S")

    # Step 1: Download data
    download_gpm_data(start_time, end_time)

    # Step 2: Load dataset
    ds = load_gpm_data(start_time, end_time)

    # Step 3: Label storms
    xr_obj = label_storms(ds)

    # Step 4: Extract patches
    label_isel_dict = extract_patches(xr_obj)

    # Step 5: Compute patch statistics
    patch_statistics = compute_patch_statistics(ds, label_isel_dict)

    # Step 6: Save results
    output_path = os.path.join(OUTPUT_DIR, "large_patch_statistics.parquet")
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating missing directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results(patch_statistics, output_path)


# Global variables
VARIABLES = [
    "sunLocalTime",
    "airTemperature",
    "precipRate",
    "paramDSD",
    "zFactorFinal",
    "zFactorMeasured",
    "precipRateNearSurface",
    "precipRateESurface",
    "precipRateESurface2",
    "zFactorFinalESurface",
    "zFactorFinalNearSurface",
    "heightZeroDeg",
    "binEchoBottom",
    "landSurfaceType",
]

if __name__ == "__main__":
    main()