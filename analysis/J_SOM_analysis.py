#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load trained SOM, assign BMUs, and visualize cluster properties.

@author: shadya
"""
#%% IMPORTS
import os
import sys
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import itertools

# Import local functions
PACKAGE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.experiments import get_experiment_info, load_som
from gpm_storm.som.io import (
    sample_node_datasets,
    create_som_sample_ds_array,
    create_som_df_array,
    create_som_df_features_stats,
    create_dask_cluster,
)
from gpm_storm.som.plot import (
    plot_images,
    plot_som_array_datasets,
    plot_som_feature_statistics,
)

# CONFIGURATION
PARALLEL = False
FILEPATH = os.path.expanduser("~/gpm_storm/data/largest_patch_statistics.parquet")
SOM_DIR = os.path.expanduser("~/gpm_storm/script")  # Update if needed
FIGS_DIR = os.path.expanduser("~/gpm_storm/figs")
SOM_NAME = "zonal_SOM"  # Change for different experiments
VARIABLE = "precipRateNearSurface"
NUM_IMAGES = 25
NCOLS = 5

def load_and_preprocess_data(filepath, som_name):
    """Load the dataset, drop NaNs, and return the processed DataFrame."""
    df = pd.read_parquet(filepath)
    info_dict = get_experiment_info(som_name)
    features = info_dict["features"]

    # Drop rows with NaN values in selected features
    df = df.dropna(subset=features)

    return df, info_dict


def assign_bmus(df, som):
    """
    Assign Best Matching Units (BMUs) to the DataFrame.
    """
    bmus = som.bmus  # Get BMU indices
    df["row"], df["col"] = bmus[:, 0], bmus[:, 1]

    # Check for missing row-col combinations
    row_values = range(5)  # Assuming 5 rows (0 to 4)
    col_values = range(5)  # Assuming 5 columns (0 to 4)

    expected_combinations = set(itertools.product(row_values, col_values))
    actual_combinations = set(zip(df["row"], df["col"]))
    missing_combinations = expected_combinations - actual_combinations

    if missing_combinations:
        print(f"⚠️ Missing {len(missing_combinations)} (row, col) combinations:")
        print(missing_combinations)
    else:
        print("✅ All row-column combinations are present!")

    # Save updated DataFrame with BMUs
    new_filepath = os.path.expanduser("~/gpm_storm/data/largest_patch_statisticss_with_bmus.parquet")
    df.to_parquet(new_filepath)
    print(f"Updated DataFrame saved to: {new_filepath}")

    return df, missing_combinations



def plot_som_grid(arr_df, figs_som_dir):
    """Plot and save the SOM grid with sample images."""
    arr_ds = create_som_sample_ds_array(arr_df, variables=VARIABLE, parallel=PARALLEL)

    img_fpath = os.path.join(figs_som_dir, "som_grid_samples.png")
    fig = plot_som_array_datasets(arr_ds, figsize=(5, 5), variable=VARIABLE)
    fig.tight_layout()
    fig.savefig(img_fpath)
    plt.close(fig)  # Free memory


def plot_som_node_samples(arr_df, df, n_rows, n_columns, figs_som_dir):
    """Generate and save node sample images and maps."""
    for row in range(n_rows):
        for col in range(n_columns):
            img_fpath = os.path.join(figs_som_dir, f"node_{row}_{col}_samples.png")
            img_fpath_map = os.path.join(figs_som_dir, f"node_{row}_{col}_map.png")

            df_node = arr_df[row, col]
            list_ds = sample_node_datasets(df_node, num_images=NUM_IMAGES, variables=VARIABLE, parallel=PARALLEL)

            # Plot sample images
            fig = plot_images(list_ds, ncols=NCOLS, figsize=(15, 15), variable=VARIABLE)
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


def plot_feature_statistics(df):
    """Plot SOM feature statistics."""
    df_stats = create_som_df_features_stats(df)
    fig = plot_som_feature_statistics(df_stats, feature="precipitation_average")
    plt.show()

#%%
def main():
    """Main function to run the SOM analysis pipeline."""


if __name__ == "__main__":
    main()
# %%

if PARALLEL:
        create_dask_cluster()

figs_som_dir = os.path.join(FIGS_DIR, SOM_NAME)
os.makedirs(figs_som_dir, exist_ok=True)

# Load and preprocess data
df, info_dict = load_and_preprocess_data(FILEPATH, SOM_NAME)
n_rows, n_columns = info_dict["som_grid_size"]

# Load trained SOM
som = load_som(som_dir=SOM_DIR, som_name=SOM_NAME)

# Assign BMUs
df, missing_combinations = assign_bmus(df, som)

# Create SOM node-based dataframes
arr_df = create_som_df_array(som=som, df=df)



# Plot SOM grid
#plot_som_grid(arr_df, figs_som_dir)

# Plot SOM node samples and maps
#plot_som_node_samples(arr_df, df, n_rows, n_columns, figs_som_dir)

# Plot feature statistics
#plot_feature_statistics(df)

# %%
som_shape = arr_df.shape
arr_ds = np.empty(som_shape, dtype=object)
# %%
import random
from gpm_storm.som.io import _open_sample_dataset

for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            # Extract images for each cell in the SOM
            df_node = arr_df[row, col]
            # Select valid random index
            index = random.randint(0, len(df_node) - 1)
            # Open dataset
            ds = _open_sample_dataset(df_node, index=index, variables=VARIABLE)
            # Add the dataset to the arrays
            arr_ds[row, col] = ds
#%%
def _get_patch_image(img): 
    max_value_position = np.unravel_index(np.argmax(img), img.shape)
    center_y, center_x = max_value_position
    if center_x < 25:
        img = img[:, 0:49]
    elif (img.shape[1] - center_x) > 25:
        start_x = center_x - 24
        end_x = center_x + 25
        img = img[:, start_x:end_x]
    else: 
        img = img[:, -49:]
    return img 
def _remove_axis(ax): 
    ax.set_title("")  # Set title to an empty string
    ax.set_xlabel("")  # Set xlabel to an empty string
    ax.set_ylabel("")  # Set ylabel to an empty string
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# %%
img_fpath = os.path.join(figs_som_dir, "som_grid_samples.png")
nrows, ncols = arr_ds.shape
cbar_kwargs = {"shrink": 0.8, "aspect": 20}
plot_kwargs = {"cmap": "viridis", "interpolation": "nearest"}
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,10))
fig.subplots_adjust(0,0,1,1, wspace=0, hspace=0)
for i in range(nrows):
        for j in range(ncols):
            ax = axes[i,j]
            da = arr_ds[i,j][VARIABLE]
            img = _get_patch_image(da.data)
            ax.imshow(img, **plot_kwargs)
            _remove_axis(ax)

#fig = plot_som_array_datasets(arr_ds, figsize=(5, 5), variable=VARIABLE)
#fig.tight_layout()
#fig.savefig(img_fpath)
#plt.close(fig)