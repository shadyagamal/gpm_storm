#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform spatial analysis and partitioning of storm data.

@author: shadya
"""

#%% Imports
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from gpm.bucket.partitioning import LonLatPartitioning
from gpm.visualization import plot_cartopy_background, plot_colorbar # type: ignore

#%% Configuration
FILE_PATH = os.path.expanduser("~/gpm_storm/data/patch_statistics_with_bmus.parquet")
OUTPUT_XARRAY = os.path.expanduser("~/gpm_storm/data/partitioned_data.nc")
BIN_SIZE = 0.1  # Size for geographic partitioning
FIGS_DIR = os.path.expanduser("~/gpm_storm/figs")

# Ensure output directory exists
os.makedirs(FIGS_DIR, exist_ok=True)


#%% Functions
def load_data(filepath):
    """Load parquet file into a Pandas DataFrame."""
    print(f"Loading data from: {filepath}")
    df = pd.read_parquet(filepath)
    return df


def compute_track_length(df):
    """Compute track length as the difference between along track start and end."""
    df["length_track"] = df["along_track_end"] - df["along_track_start"]
    return df


def apply_spatial_partitioning(df, bin_size=0.1):
    """Apply spatial binning using LonLatPartitioning."""
    partitioning = LonLatPartitioning(size=bin_size)
    df = partitioning.add_labels(df, x="lon", y="lat")
    df = partitioning.add_centroids(df, x="lon", y="lat")
    ds = partitioning.to_xarray(df)
    print("✅ Spatial partitioning applied.")
    
    # Save as NetCDF for further analysis
    ds.to_netcdf(OUTPUT_XARRAY)
    print(f"✅ Saved partitioned data as NetCDF: {OUTPUT_XARRAY}")
    return df, ds


def compute_binned_statistics(df):
    """Compute count and median statistics per latitude/longitude bin."""
    df["lon_bin"] = df["lon"].round(1)
    df["lat_bin"] = df["lat"].round(1)
    grouped_df = df.groupby(["lon_bin", "lat_bin"])
    binned_df = grouped_df.agg(["count", "median"])
    return binned_df


def visualize_spatial_data(df):
    """Generate and save plots for spatial data visualization."""
    df_subset = df[(df["row"] == 0) & (df["col"] == 8)]
    lon = df_subset["lon"].values
    lat = df_subset["lat"].values
    value = df_subset["echodepth30_mean"]

    # Scatter plot with fixed color
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    plot_cartopy_background(ax)
    ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c="orange", s=2)
    fig.savefig(os.path.join(FIGS_DIR, "scatter_fixed.png"))
    plt.close(fig)

    # Scatter plot with value color mapping
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    plot_cartopy_background(ax)
    p = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=value, s=4, cmap="Spectral", vmax=5000)
    plot_colorbar(p=p, ax=ax)
    fig.savefig(os.path.join(FIGS_DIR, "scatter_value.png"))
    plt.close(fig)

    print("✅ Saved spatial visualizations.")


#%% Main Execution
def main():
    """Main function to run the spatial analysis."""
    df = load_data(FILE_PATH)

    # Compute track length
    df = compute_track_length(df)

    # Apply spatial binning
    df, ds = apply_spatial_partitioning(df, bin_size=BIN_SIZE)

    # Compute statistics
    binned_df = compute_binned_statistics(df)
    print("✅ Computed binned statistics.")

    # Visualize results
    visualize_spatial_data(df)


if __name__ == "__main__":
    main()

# %%
