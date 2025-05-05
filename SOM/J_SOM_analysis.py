#!/usr/bin/env python3
"""
Load trained SOM, assign BMUs, and visualize cluster properties.

@author: shadya
"""
# IMPORTS
import itertools
import os
import sys
import pycolorbar
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import pandas as pd
import random
import xarray as xr
import glob
import gpm
from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.experiments import get_experiment_info, load_som
from gpm_storm.som.io import (
    create_dask_cluster,
    create_som_df_array,
    create_som_df_features_stats,
    create_som_sample_ds_array,
    sample_node_datasets,
)
from gpm_storm.som.plot import (
    plot_images,
)

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


def build_som_patch_array(df, som, zarr_directory, variable="precipRateNearSurface"):
    arr_df = create_som_df_array(som=som, df=df)
    som_shape = arr_df.shape
    arr_ds = np.empty(som_shape, dtype=object)

    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            df_node = arr_df[row, col]
            if len(df_node) == 0:
                continue

            index = random.randint(0, len(df_node) - 1)
            patch_row = df_node.iloc[index]
            granule_id = str(patch_row["gpm_granule_id"])
            patch_id = patch_row["patch_id"]
            time = pd.to_datetime(patch_row["time"])
            year, month = time.year, time.month

            search_path = os.path.join(zarr_directory, f"{year:04d}/{month:02d}", "*.zarr")
            zarr_files = glob.glob(search_path)

            patch_ds = None
            for zarr_file in zarr_files:
                if granule_id in os.path.basename(zarr_file):
                    ds = xr.open_zarr(zarr_file)
                    patch_ds = ds.isel(patch=patch_id)
                    break

            if patch_ds is None:
                print(f"⚠️ No match for SOM ({row},{col}) granule {granule_id}")
                continue

            arr_ds[row, col] = patch_ds

    return arr_df, arr_ds



def plot_som_grid_samples(arr_ds, save_dir, cbar_kwargs={}, **plot_kwargs):
    os.makedirs(save_dir, exist_ok=True)
    som_shape = arr_ds.shape

    fig, axes = plt.subplots(*som_shape, figsize=(10, 10))
    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            ax = axes[row, col]
            ax.axis("off")
            ds = arr_ds[row, col]
            if ds is not None:
                da = ds["precipRateNearSurface"]
                plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(da.name, user_plot_kwargs=plot_kwargs,
                                                               user_cbar_kwargs=cbar_kwargs)
                da.plot.imshow(
                    ax=ax,
                    add_colorbar=False,
                    add_labels=False,
                    interpolation="nearest",
                    **plot_kwargs,
                )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "som_grid_samples_clean.png"), dpi=300)
    plt.close(fig)
    return None

def plot_images_new(list_ds, ncols=5, figsize=(15, 5),
                    variable="precipRateNearSurface"):
    num_images = len(list_ds)
    plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(variable)
    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(np.ceil(num_images / ncols))  # Adjust as needed
    num_cols = min(num_images, ncols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
    
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i < num_images:
            da = list_ds[i][variable]
            da.plot.imshow(
                ax=ax,
                add_colorbar=False,
                add_labels=False,
                interpolation="nearest",
                **plot_kwargs,
            )
    return fig



def plot_node_samples_and_maps(arr_df, df, zarr_directory, save_dir, variable="precipRateNearSurface", 
                                num_images=25, n_rows=10, n_cols=10):
    os.makedirs(save_dir, exist_ok=True)
    for row in range(n_rows):
        for col in range(n_cols):
            df_node = arr_df[row, col]
            if len(df_node) < num_images:
                continue  # Skip if not enough samples

            random_indices = random.sample(range(len(df_node)), num_images)
            list_ds = []

            for index in random_indices:
                patch_row = df_node.iloc[index]
                patch_ds = None
                zarr_file, patch_ds = find_zarr_file_for_patch(patch_row, zarr_directory, filename_pattern="*.zarr")
                
    
                if patch_ds:
                    list_ds.append(patch_ds)

            if list_ds:
                fig = plot_images_new(list_ds, figsize=(15, 15), ncols=5, variable=variable) 
                fig.tight_layout()
                img_path = os.path.join(save_dir, f"node_{row}_{col}_samples.png")
                fig.savefig(img_path)
                plt.close(fig)

            df_subset = df[(df["row"] == row) & (df["col"] == col)].copy()

            if not df_subset.empty:
                df_subset["time"] = pd.to_datetime(df_subset["time"])
                df_subset["month"] = df_subset["time"].dt.month

                fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                plot_cartopy_background(ax)
                sc = ax.scatter(
                    df_subset["lon"], df_subset["lat"], 
                    transform=ccrs.PlateCarree(), c=df_subset["month"], s=2
                )
    
                pycolorbar.plot_colorbar(p=sc, ax=ax, orientation="vertical", label="month") #pad=0.02))
                # cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
                # cbar.set_label("Month")

                map_path = os.path.join(save_dir, f"node_{row}_{col}_map.png")
                fig.savefig(map_path)
                plt.close(fig)

            
            
def plot_mean_variable_per_node(arr_df, save_dir, variable="P_mean"):
    mean_values = np.full((10, 10), np.nan)

    for row in range(10):
        for col in range(10):
            df_node = arr_df[row, col]
            if not df_node.empty:
                mean_val = df_node[variable].mean()
                mean_values[row, col] = mean_val

    plt.figure(figsize=(8, 8))
    cmap = plt.cm.viridis
    masked_array = np.ma.masked_invalid(mean_values)

    plt.imshow(masked_array, cmap=cmap, origin="upper")
    cbar = plt.colorbar()
    cbar.set_label(f"Mean {variable}")
    plt.title(f"Mean {variable} per SOM Node")
    plt.xlabel("SOM Column")
    plt.ylabel("SOM Row")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.grid(False)

    mean_path = os.path.join(save_dir, f"som_mean_{variable}.png")
    plt.savefig(mean_path, dpi=300)
    plt.show()
    return None


# --- Config ---
filepath = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
df = pd.read_parquet(filepath)
som_dir = os.path.expanduser("~/gpm_storm/SOM/trained_soms/")  
som_name = "Test_SOM"
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}")
os.makedirs(figs_dir, exist_ok=True)
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"

# --- Load ---
df_bmu = pd.read_parquet(bmu_dir)
som = load_som(som_dir=som_dir, som_name=som_name)
bmus = som.bmus

row_values = range(10)  
col_values = range(10)  
expected_combinations = set(itertools.product(row_values, col_values))
actual_combinations = set(zip(df_bmu["row"], df_bmu["col"], strict=False))
missing_combinations = expected_combinations - actual_combinations
if missing_combinations:
    print(f"Missing nodes: {missing_combinations}")
else:
    print("No missing (row, col) combinations.\n")


# --- Build SOM Patch Arrays ---
arr_df, arr_ds = build_som_patch_array(df_bmu, som, zarr_directory)

# --- Plot Grid of Samples ---
plot_som_grid_samples(
    arr_ds, save_dir=figs_dir, 
    # cmap="turbo", 
    # norm=colors.LogNorm(vmin=0.01, vmax=300)
)

# --- Plot Node Samples and Maps ---
plot_node_samples_and_maps(
    arr_df, df_bmu, zarr_directory, 
    save_dir=figs_dir, 
    variable="precipRateNearSurface", 
    num_images=25
)

# --- Plot Mean Variable per Node ---
plot_mean_variable_per_node(
    arr_df, save_dir=figs_dir, variable="P_mean"
)

