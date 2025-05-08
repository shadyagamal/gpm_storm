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
from collections import defaultdict
import seaborn as sns
from sklearn.feature_selection import f_regression

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

def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"
    
season_palette = {
    "Winter": "blue",
    "Spring": "green",
    "Summer": "red",
    "Autumn": "orange"
}    

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
                df_subset["season"] = df_subset["month"].apply(month_to_season)
               

                fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={"projection": ccrs.PlateCarree()})
                plot_cartopy_background(ax)
                sns.scatterplot(data=df_subset, x="lon", y="lat", palette=season_palette, 
                                     hue="season", s=10, edgecolor=None)
                ax.legend(title="Season", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
                map_path = os.path.join(save_dir, f"node_{row}_{col}_map.png")
                fig.savefig(map_path)
                plt.close(fig)

            
            
def plot_mean_variable_per_node(arr_df, save_dir, variable="P_mean", high_q=0.99, low_q=0.01, fig=True):
    mean_values = np.full((10, 10), np.nan)

    for row in range(10):
        for col in range(10):
            df_node = arr_df[row, col]
            if not df_node.empty:
                mean_val = df_node[variable].mean()
                mean_values[row, col] = mean_val
                
    valid_means = mean_values[~np.isnan(mean_values)]
    high_thresh = np.quantile(valid_means, high_q)
    low_thresh = np.quantile(valid_means, low_q)

    for row in range(10):
        for col in range(10):
            val = mean_values[row, col]
            if np.isnan(val):
                continue
            if val >= high_thresh:
                node_characteristics[(row, col)].append(f"high {variable}")
            if val <= low_thresh:
                node_characteristics[(row, col)].append(f"low {variable}")
    if fig==True:
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
        plt.close()
    return None


# --- Config ---
# filepath0 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
# filepath1 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet") 
# df0 = pd.read_parquet(filepath0) 
# df1 = pd.read_parquet(filepath1) 
# df = pd.concat([df0,df1])
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
som_name = "SOM_Pmean_>_1_uniform"
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}")
os.makedirs(figs_dir, exist_ok=True)
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"

# --- Load ---
df_bmu = pd.read_parquet(bmu_dir)
som = load_som(som_dir=som_dir, som_name=som_name)
bmus = som.bmus

row_values = range(som._n_rows)
col_values = range(som._n_columns)

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
node_characteristics = defaultdict(list)
for col in df_bmu.columns[:134]:
    plot_mean_variable_per_node(
        arr_df, save_dir=figs_dir, variable=col, fig=True)

# --- Grouped Analysis ---
grouped = df_bmu.groupby(['row', 'col'])

summary = grouped.mean(numeric_only=True)
summary_std = grouped.std(numeric_only=True)
counts = grouped.size().unstack(fill_value=0)


# Per node
node_df = df_bmu[(df_bmu['row'] == 9) & (df_bmu['col'] == 4)]
print(f"{len(node_df)} events in node (3, 9)")
vars = node_df.columns[:134]

df_pivot = node_df.pivot_table(index=['lat', 'lon'], values=vars).dropna(axis=1)
f_lat, _ = f_regression(df_pivot, df_pivot.index.get_level_values('lat'))
f_lon, _ = f_regression(df_pivot, df_pivot.index.get_level_values('lon'))
spatial_dependency = pd.DataFrame({'var': df_pivot.columns, 'lat_score': f_lat, 'lon_score': f_lon})
spatial_dependency['combined_score'] = spatial_dependency['lat_score'] + spatial_dependency['lon_score']

top_vars = spatial_dependency.sort_values('combined_score', ascending=False).head(8)['var']

for var in top_vars[3:]:
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    plot_cartopy_background(ax)
    sns.scatterplot(data=node_df, x="lon", y="lat", palette="viridis", 
                         hue=var, s=10, edgecolor=None)
    ax.legend(title=f"{var}", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.title(f'Spatial distribution of {var}')
    plt.show()
    



features = ["P_mean", "P_max", "P_count", "REFC_mean", "REFCH_mean", "CC_30_count", "lon", "lat"]
for var in features:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(df_bmu[var], label="All data", linewidth=2)
    sns.kdeplot(node_df[var], label="Node (12,5)", linewidth=2)
    plt.title(f"Distribution of {var}")
    plt.legend()
    plt.tight_layout()
    plt.show()
