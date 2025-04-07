#!/usr/bin/env python3
"""
Load trained SOM, assign BMUs, and visualize cluster properties.

@author: shadya
"""
# IMPORTS
import itertools
import os
import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import xarray as xr
import glob
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
    plot_som_array_datasets,
    plot_som_feature_statistics,
)
from matplotlib import colors
from pyart.graph import *
cmap = "NWSRef"



# CONFIGURATION
parallel = False
filepath = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
SOM_dir = os.path.expanduser("~/gpm_storm/scripts")  # Update if needed
figs_dir = os.path.expanduser("~/gpm_storm/figs")
SOM_name= "zonal_SOM"  # Change for different experiments
variable = "precipRateNearSurface"
Nimages = 25
Ncols = 5
figsize=(10, 10)
figs_som_dir = os.path.join(figs_dir, SOM_name)
os.makedirs(figs_som_dir, exist_ok=True)

if parallel: 
    create_dask_cluster() 
    
df = pd.read_parquet(filepath)
vars = df.columns[0:-9]
info_dict = get_experiment_info(SOM_name)
features = info_dict["features"]
df = df.dropna(subset=vars)

som = load_som(som_dir=SOM_dir, som_name=SOM_name)
bmus = som.bmus  # Get BMU indices
df["row"], df["col"] = bmus[:, 0], bmus[:, 1]

# Check for missing row-col combinations
row_values = range(10)  
col_values = range(10)  
expected_combinations = set(itertools.product(row_values, col_values))
actual_combinations = set(zip(df["row"], df["col"], strict=False))
missing_combinations = expected_combinations - actual_combinations

# Save updated DataFrame with BMUs
new_filepath = os.path.expanduser("~/gpm_storm/data/merged_data_total_0_with_bmus.parquet")
df.to_parquet(new_filepath)

arr_df = create_som_df_array(som=som, df=df)

som_shape = arr_df.shape
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"  
arr_ds = np.empty(som_shape, dtype=object)

for row in range(som_shape[0]):
    for col in range(som_shape[1]):
        df_node = arr_df[row, col]
        if len(df_node) == 0:
            continue  # No patch for this node

        # Select a random patch from this node
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



# fig = plot_som_array_datasets(arr_ds, figsize=figsize, variable=variable)
# fig.tight_layout()
# img_fpath = os.path.join(figs_som_dir, "som_grid_samples_gpm_plot_image.png")
# plt.savefig(img_fpath, dpi=300)
# plt.show()
# fig.close()


cmap = "Wild25"
norm = colors.LogNorm(vmin=0.01, vmax=160)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for row in range(10):
    for col in range(10):
        ax = axes[row, col]
        ax.axis("off")

        ds = arr_ds[row, col]
        if ds is not None:
            da = ds["precipRateNearSurface"]
            da.plot.imshow(
                ax=ax,
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                add_labels=False,
                interpolation="nearest"
            )

plt.tight_layout()
img_fpath = os.path.join(figs_som_dir, "som_grid_samples_clean.png")
plt.savefig(img_fpath, dpi=300)
plt.show()

### Plot SOM node samples 
variable = "precipRateNearSurface"
num_images = 25
n_rows = 5
n_columns = 5
figsize = (15, 15)

for row in range(n_rows):
    for col in range(n_columns):
        img_fpath = os.path.join(figs_som_dir, f"node_{row}_{col}_samples.png")
        img_fpath_map = os.path.join(figs_som_dir, f"node_{row}_{col}_map.png")
        df_node = arr_df[row, col]
        random_indices = random.sample(range(len(df_node)), num_images)
        list_ds = []
        for index in random_indices:
            print(index)
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
            list_ds.append(patch_ds)
 
        
        # Plot sample images
        fig = plot_images(list_ds, ncols=Ncols, figsize=(15, 15), variable=variable)
        fig.tight_layout()
        fig.savefig(img_fpath)
        plt.close(fig)  # Free memory

        # Filter dataset for this node
        df_subset = df[(df["row"] == row) & (df["col"] == col)].copy()

        if not df_subset.empty:
            df_subset["time"] = pd.to_datetime(df_subset["time"])
            df_subset["month"] = df_subset["time"].dt.month

            lon, lat = df_subset["lon"].values, df_subset["lat"].values

            # Plot geographic distribution
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
            plot_cartopy_background(ax)
            sc = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=df_subset["month"], s=2)

            cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cbar.set_label("Month")

            fig.savefig(img_fpath_map)
            plt.close(fig)  # Free memory

# df_stats = create_som_df_features_stats(df)
# fig = plot_som_feature_statistics(df_stats, feature='precipitation_average')

# def plot_feature_statistics(df):
#     """Plot SOM feature statistics."""
#     df_stats = create_som_df_features_stats(df)
#     fig = plot_som_feature_statistics(df_stats, feature="precipitation_average")
#     plt.show()


# # 
# def main():
#     """Main function to run the SOM analysis pipeline."""


# if __name__ == "__main__":
#     main()



# # Plot SOM grid
# # plot_som_grid(arr_df, figs_som_dir)

# # Plot SOM node samples and maps
# # plot_som_node_samples(arr_df, df, n_rows, n_columns, figs_som_dir)

# # Plot feature statistics
# # plot_feature_statistics(df)


# # 
# def _get_patch_image(img):
#     max_value_position = np.unravel_index(np.argmax(img), img.shape)
#     center_y, center_x = max_value_position
#     if center_x < 25:
#         img = img[:, 0:49]
#     elif (img.shape[1] - center_x) > 25:
#         start_x = center_x - 24
#         end_x = center_x + 25
#         img = img[:, start_x:end_x]
#     else:
#         img = img[:, -49:]
#     return img


# def _remove_axis(ax):
#     ax.set_title("")  # Set title to an empty string
#     ax.set_xlabel("")  # Set xlabel to an empty string
#     ax.set_ylabel("")  # Set ylabel to an empty string
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)


# # 
# # img_fpath = os.path.join(figs_som_dir, "som_grid_samples.png")
# # nrows, ncols = arr_ds.shape
# # cbar_kwargs = {"shrink": 0.8, "aspect": 20}
# # plot_kwargs = {"cmap": "viridis", "interpolation": "nearest"}
# # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
# # fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
# # for i in range(nrows):
# #     for j in range(ncols):
# #         ax = axes[i, j]
# #         da = arr_ds[i, j][VARIABLE]
# #         img = _get_patch_image(da.data)
# #         ax.imshow(img, **plot_kwargs)
# #         _remove_axis(ax)

# # fig = plot_som_array_datasets(arr_ds, figsize=(5, 5), variable=VARIABLE)
# # fig.tight_layout()
# # fig.savefig(img_fpath)
# # plt.close(fig)
