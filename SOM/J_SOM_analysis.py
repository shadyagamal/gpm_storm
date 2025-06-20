#!/usr/bin/env python3
"""
Load trained SOM, assign BMUs, and visualize cluster properties.

@author: shadya
"""
# --- IMPORTS ---
import itertools
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import xarray as xr
import glob
import gpm
from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.experiments import load_som
from gpm_storm.som.io import create_som_df_array
import matplotlib.patches as patches
from collections import defaultdict
import seaborn as sns
from sklearn.feature_selection import f_regression
from kgcpy import *
import datetime
from matplotlib.colors import LogNorm, Normalize
from xhistogram.xarray import histogram
from tqdm import tqdm
import pickle 

# --- CONSTANTS ---
SEASON_PALETTE = {
    "Winter": "blue",
    "Spring": "green",
    "Summer": "red",
    "Autumn": "orange"
}

# --- UTILITY FUNCTIONS ---
def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else: 
        return "Autumn"
  
    
def detect_missing_combos(som):
    row_values = range(som._n_rows)
    col_values = range(som._n_columns)
    expected_combinations = set(itertools.product(row_values, col_values))
    actual_combinations = set(zip(df_bmu["row"], df_bmu["col"], strict=False))
    missing_combinations = expected_combinations - actual_combinations
    if missing_combinations:
        print(f"Missing nodes: {missing_combinations}")
    else:
        print("No missing (row, col) combinations.\n")
        
def find_zarr_file_for_patch(row, zarr_directory, filename_pattern="*.zarr"):
    granule_id, patch_id = str(row["gpm_granule_id"]), row["patch_id"]
    time = pd.to_datetime(row["time"])
    
    for offset in [0, -1]:
        t = time + pd.DateOffset(months=offset)
        path = os.path.join(zarr_directory, f"{t.year:04d}/{t.month:02d}", filename_pattern)
        for zarr_file in glob.glob(path):
            if granule_id in os.path.basename(zarr_file):
                ds = xr.open_zarr(zarr_file)
                if patch_id < ds.sizes["patch"]:
                    return zarr_file, ds.isel(patch=patch_id)
    print(f"No Zarr for granule_id {granule_id} at {time.strftime('%Y-%m')}")
    return None, None

# --- SOM HANDLING ---
def build_som_patch_array(df, som, zarr_directory, variable="precipRateNearSurface"):
    arr_df = create_som_df_array(som=som, df=df)
    arr_ds = np.empty(arr_df.shape, dtype=object)

    for row in range(arr_df.shape[0]):
        for col in range(arr_df.shape[1]):
            df_node = arr_df[row, col]
            if not len(df_node):
                continue
            sample = df_node.sample(1).iloc[0]
            _, patch_ds = find_zarr_file_for_patch(sample, zarr_directory)
            if patch_ds is not None:
                arr_ds[row, col] = patch_ds

    return arr_df, arr_ds

# --- PLOTTING ---

def plot_som_grid_samples(arr_ds, save_dir, variable="precipRateNearSurface", **plot_kwargs):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(*arr_ds.shape, figsize=(15, 15), squeeze=False)

    for row in range(arr_ds.shape[0]):
        for col in range(arr_ds.shape[1]):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            ds = arr_ds[row, col]

            if ds is not None:
                da = ds[variable]
                plot_kws, _ = gpm.get_plot_kwargs(da.name, user_plot_kwargs=plot_kwargs)
                da.plot.imshow(ax=ax, add_colorbar=False, add_labels=False, interpolation="nearest", **plot_kws)
                ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, edgecolor='black', facecolor='none'))

            if col == 0:
                ax.text(-0.5, 0.5, str(row), transform=ax.transAxes, ha='left', va='center', fontsize=20)
            if row == arr_ds.shape[0] - 1:
                ax.text(0.5, -0.5, str(col), transform=ax.transAxes, ha='center', va='bottom', fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "som_grid_samples_clean.png"), dpi=300)
    plt.close(fig)
    
def plot_images_grid(list_ds, variable="precipRateNearSurface", ncols=5, figsize=(15, 15)):
    n = len(list_ds)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    plot_kws, _ = gpm.get_plot_kwargs(variable)
    
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i < n:
            da = list_ds[i][variable]
            da.plot.imshow(ax=ax, add_colorbar=False, add_labels=False, interpolation="nearest", **plot_kws)
    return fig


def plot_node_samples_and_maps(arr_df, df, zarr_directory, save_dir, variable="precipRateNearSurface", num_images=25):
    os.makedirs(save_dir, exist_ok=True)
    for row in range(arr_df.shape[0]):
        for col in range(arr_df.shape[1]):
            df_node = arr_df[row, col]
            if len(df_node) < num_images:
                continue

            list_ds = []
            for idx in random.sample(range(len(df_node)), num_images):
                patch_row = df_node.iloc[idx]
                _, patch_ds = find_zarr_file_for_patch(patch_row, zarr_directory)
                if patch_ds:
                    list_ds.append(patch_ds)

            if list_ds:
                fig = plot_images_grid(list_ds, variable=variable)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, f"node_{row}_{col}_samples.png"))
                plt.close(fig)

            df_subset = df[(df["row"] == row) & (df["col"] == col)].copy()
            if not df_subset.empty:
                df_subset["time"] = pd.to_datetime(df_subset["time"])
                df_subset["month"] = df_subset["time"].dt.month
                df_subset["Season"] = df_subset.apply(lambda x: month_to_season(x["month"]), axis=1)

                fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={"projection": ccrs.PlateCarree()})
                plot_cartopy_background(ax)
                sns.scatterplot(data=df_subset, x="lon", y="lat", hue="Season", palette=SEASON_PALETTE, ax=ax)
                ax.set_title(f"Node ({row}, {col}) - Patch Locations by Season")
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                fig.savefig(os.path.join(save_dir, f"node_{row}_{col}_map.png"))
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
        plt.close(fig)
    return None


# --- Config ---
som_name = "SOM_Pmean_>_1_with_random_init"  
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}")
kde_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/KDE")
res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/0_Results")
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"

os.makedirs(figs_dir, exist_ok=True)
os.makedirs(kde_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
    
    
# LOAD
with open(os.path.join(res_dir,"cfad_totals.pkl"), "rb") as f:
    cfad_totals = pickle.load(f)
with open(os.path.join(res_dir,"cfad_convs.pkl"), "rb") as f:
    cfad_convs = pickle.load(f)
with open(os.path.join(res_dir,"cfad_strats.pkl"), "rb") as f:
    cfad_strats = pickle.load(f)
with open(os.path.join(res_dir,"cfad_total_tops.pkl"), "rb") as f:
    cfad_total_tops = pickle.load(f)
with open(os.path.join(res_dir,"cfad_others.pkl"), "rb") as f:
    cfad_others = pickle.load(f)
with open(os.path.join(res_dir,"wcc_flagss.pkl"), "rb") as f:
    wcc_flags = pickle.load(f)   

# --- Load ---
df_bmu = pd.read_parquet(bmu_dir)
som = load_som(som_dir=som_dir, som_name=som_name)
n_rows = som._n_rows
n_cols = som._n_columns
detect_missing_combos(som)
arr_df, arr_ds = build_som_patch_array(df_bmu, som, zarr_directory)

# --- Plot Grid of Samples ---
plot_som_grid_samples(arr_ds, save_dir=figs_dir)

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
    
node_dict_path = os.path.join(res_dir, "node_characteristics.pkl")
with open(node_dict_path, 'wb') as f:
    pickle.dump(node_characteristics, f)

with open(node_dict_path, 'rb') as f:
    loaded_dict = pickle.load(f)
    
# --- Grouped Analysis ---
grouped = df_bmu.groupby(['row', 'col'])
grouped_summary = grouped.mean(numeric_only=True)
grouped_counts = grouped.size().unstack(fill_value=0)


# --- Node Analysis ---
target_node = (0,0)
node_df = df_bmu[(df_bmu["row"] == target_node[0]) & (df_bmu["col"] == target_node[1])].copy()
print(f"{len(node_df)} events in node {target_node}")
node_df["time"] = pd.to_datetime(node_df["time"])
node_df["month"] = node_df["time"].dt.month
node_df["Season"] = node_df.apply(lambda x: month_to_season(x["month"]), axis=1)

for var in df_bmu.columns[:134]:
    if var in ["lat", "row", "col"]:
        continue
    plt.figure(figsize=(6, 4))
    try:
        sns.lmplot(
            x="lat", y=var, x_bins=np.arange(-90, 100, 10), data=node_df, 
            scatter=True, lowess=True, hue="Season", ci=95)

        plt.xlabel("Latitude")
        plt.ylabel(var)
        plt.title(f"{var} vs Latitude (Smoothed)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipped {var} due to error: {e}")
        

# --- Spatial Dependency (F-stat) ---
def compute_spatial_dependency(df):
    vars_ = df.columns[:134]
    df_pivot = df.pivot_table(index=["lat", "lon"], values=vars_).dropna(axis=1)
    f_lat, _ = f_regression(df_pivot, df_pivot.index.get_level_values("lat"))
    f_lon, _ = f_regression(df_pivot, df_pivot.index.get_level_values("lon"))
    dep_df = pd.DataFrame({"var": df_pivot.columns, "lat_score": f_lat, "lon_score": f_lon})
    dep_df["combined_score"] = dep_df["lat_score"] + dep_df["lon_score"]
    return dep_df.sort_values("combined_score", ascending=False)

top_vars = compute_spatial_dependency(node_df).head(8)["var"]

for var in top_vars[3:]:
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    plot_cartopy_background(ax)
    sns.scatterplot(data=node_df, x="lon", y="lat", palette="viridis", 
                         hue=var, s=10, edgecolor=None)
    ax.legend(title=f"{var}", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.title(f'Spatial distribution of {var}')
    plt.show()
    


# --- KDE Distributions for Selected Features ---
features = ["P_mean", "P_max", "P_count", "REFC_mean", "REFCH_mean", "CC_30_count", "lon", "lat"]
for var in features:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(df_bmu[var], label="All data", linewidth=2)
    sns.kdeplot(node_df[var], label="Node (12,5)", linewidth=2)
    plt.title(f"Distribution of {var}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- KDE Grid by Node ---
actual_combinations = list(grouped.groups.keys())

for var in df_bmu.columns[:-16]:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    for row, col in actual_combinations:
        node_data = df_bmu[(df_bmu["row"] == row) & (df_bmu["col"] == col)]
        idx = row * n_cols + col
        if idx >= len(axes): continue
        ax = axes[idx]
        if not node_data.empty:
            sns.kdeplot(node_data[var], ax=ax, linewidth=2)
        ax.set_title(f"({row},{col})")
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    plt.suptitle(f"Distributions of {var} by SOM node", fontsize=16)
    plt.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(kde_dir, f"KDE_{var}.png"))
    plt.close()
    
    
# #  --- Köppen Zone Mapping ---

# df_bmu["kg_zone"] = df_bmu.apply(lambda row: lookupCZ(row["lat"], row["lon"]), axis=1)
# zone = "As"
# zone_counts = df_bmu.groupby(["row", "col", "kg_zone"]).size().unstack(fill_value=0)
# zone_map = zone_counts[zone].unstack().fillna(0)

# plt.figure(figsize=(8, 6))
# plt.imshow(zone_map, cmap="viridis", origin="lower")
# plt.colorbar(label=f"Count of zone {zone}")
# plt.title(f"Distribution of Köppen Zone {zone} across SOM nodes")
# plt.xlabel("col")
# plt.ylabel("row")
# plt.show()

# fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={"projection": ccrs.PlateCarree()})
# plot_cartopy_background(ax)
# sns.scatterplot(data=df_bmu, x="lon", y="lat", hue="kg_zone", s=10, edgecolor=None, ax=ax)
# ax.legend(title="KG - Zone", bbox_to_anchor=(1.01, 1), loc="upper left")
# plt.show()






# ---------------

# # WCC - Wide Convective Core
# # Contiguous 3-D convective echo objects exceeding either the moderate or strong threshold intensity whose horizontal dimensions (at some altitude) exceed a given threshold.
# # 	Moderate (30dBZ threshold):horizontal area threshold size is 800 km2 = 32 pixels (each pixel is 5x5 km) 
# # 	Strong (40dBZ threshold):the area threshold is 1000 km2 = 40 pixels 
# heights = z_ku.coords["height"].values
# target_height = sample["ETH30_max"]
# closest_idx = np.argmin(np.abs(heights - target_height))
# z_slice = z_ku.isel(height=closest_idx)
# mask_30dbz = z_slice > 30
# labels = label(mask_30dbz)
# sizes = np.bincount(labels.ravel())[1:] 
# areas_km2 = sizes * 25
# wcc_moderate = (areas_km2 >= 800).any()
# wcc_strong = (z_slice > 40).sum().item() * 25 >= 1000


df_sa_hail = df_bmu[
    (df_bmu["lon"] > -81) & 
    (df_bmu["lon"] < -35) & 
    (df_bmu["lat"] > -35) & 
    (df_bmu["lat"] < 12) & 
    (df_bmu["hail_flag"] == True)
]


# 39620
# 69140
# 90169
# 94429

sample = df_sa_hail.loc[94429]
_, patch_ds = find_zarr_file_for_patch(sample, zarr_directory)
patch_ds['precipRateNearSurface'].gpm.plot_image()

import gpm_geo
satellite="GOES16"
product="RAD"
ds_gpm_patch = patch_ds
ds_gpm_patch["gpm_granule_id"] = xr.ones_like(ds_gpm_patch["gpm_along_track_id"])*ds_gpm_patch["gpm_granule_id"] 

ds_gpm_patch = ds_gpm_patch.compute()
gpm_geo_fpath = gpm_geo.find_files_matching_acquisitions(satellite=satellite, 
                                                         product="RAD", 
                                                         timesteps=ds_gpm_patch.gpm.start_time)[0]
ds_gpm_geo = gpm_geo.open_products(gpm_geo_fpath, chunks={})

# Align GPM-GEO dataset to GPM patch
ds_gpm_geo_patch, ds_gpm_patch = gpm_geo.align_datasets(ds_gpm_geo, ds_gpm_patch, cross_track=True)

# Extract coincident GEO imagery
ds_gpm_geo_image = ds_gpm_geo_patch.gpm_geo.select_collocated_image()

# Put GPM data into memory 
ds_gpm_patch = ds_gpm_patch.compute()
ds_gpm_patch = ds_gpm_patch.sel(radar_frequency="Ku")
ds_gpm_geo_patch.gpm_geo.available_composites_names()


geo_composite = "cimss_true_color"
dummy_variable ="zFactorFinalNearSurface"
interpolation = "quadric"


##----------------------------------------------------------.
#### - Display DPR
# - Surface reflectivity    
gpm_geo.plot_composite_and_gpm(
    ds_rad=ds_gpm_geo_image,
    ds_gpm=ds_gpm_patch,
    gpm_variable="zFactorFinalNearSurface",
    geo_composite=geo_composite,
    interpolation=interpolation,
    plot_gpm=True,
    add_colorbar=True, 
    visible_colorbar=True,
)

gpm_geo.plot_composite_and_gpm(
    ds_rad=ds_gpm_geo_image,
    ds_gpm=ds_gpm_patch,
    gpm_variable="zFactorFinalNearSurface",
    geo_composite=geo_composite,
    interpolation=interpolation,
    plot_gpm=False,
    add_colorbar=True, 
    visible_colorbar=True,
)

##----------------------------------------------------------.
ds_gpm_patch
ds_gpm_geo_patch

##----------------------------------------------------------.
for i in range(ds_gpm_geo_patch.sizes["geo_acquisition"]):
    ds_gpm_geo_patch.isel(geo_acquisition=i).gpm_geo.plot("true_color")
    plt.show()
    
    
ds_gpm_geo_image.gpm_geo.plot("cimss_true_color")
ds_gpm_patch["zFactorFinalNearSurface"].gpm.plot_image()



# ------------------------

node_09 = df_bmu[(df_bmu["row"] == 0) & (df_bmu["col"] == 9)]
node_33 = df_bmu[(df_bmu["row"] == 3) & (df_bmu["col"] == 3)]


cfad_totals
cfad_convs
cfad_strats
cfad_total_tops
cfad_others
wcc_flags


for var in df_bmu.columns:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(df_bmu[var], label="All data", linewidth=2)
    sns.kdeplot(node_09[var], label="Node (0,9)", linewidth=2)
    sns.kdeplot(node_33[var], label="Node (3,3)", linewidth=2)
    plt.title(f"Distribution of {var}")
    plt.legend()
    plt.tight_layout()
    plt.show()


fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
sns.scatterplot(data=node_09, x="lon", y="lat", ax=ax, label="node (0,9)")
sns.scatterplot(data=node_33, x="lon", y="lat", ax=ax, label="node (3,3)")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()


def compute_wcc_flags(z_slice):
    pixel_area_km2=25
    
    mask_30dbz = z_slice > 30
    areas_30 = mask_30dbz.sum().item() * pixel_area_km2
    wcc_moderate = areas_30 >= 800

    mask_40dbz = z_slice > 40
    area_strong = mask_40dbz.sum().item() * pixel_area_km2
    wcc_strong = area_strong >= 1000

    return wcc_moderate, wcc_strong

wcc_flag = []
sample = node_09.iloc[5]

ds = gpm.open_dataset(
    product="2A-DPR",
    product_type="RS",
    version=7,
    start_time=sample["time"] - datetime.timedelta(minutes=1),
    end_time=sample["time"] + datetime.timedelta(minutes=1),
    scan_mode="FS",
    chunks={},
    variables=["zFactorFinal", "height", "typePrecip", "heightZeroDeg"]
    )
ds_patch = ds.gpm.sel(gpm_id=slice(sample["gpm_id_start"], sample["gpm_id_end"])).compute()
ds_patch_footprints = ds_patch.stack(footprints=["cross_track", "along_track"])


z_ku = ds_patch_footprints["zFactorFinal"].sel(radar_frequency="Ku")
height = ds_patch_footprints["height"]


heights = height.isel(footprints=0).values
target_height = np.clip(sample["ETH30_max"], heights.min(), heights.max())
closest_idx = np.argmin(np.abs(heights - target_height))
z_slice = z_ku.isel(range=closest_idx)
z_slice_2d = z_slice.unstack("footprints")
wcc_moderate, wcc_strong = compute_wcc_flags(z_slice_2d)

wcc_flag.append({
    "index": sample.name,
    "wcc_moderate": wcc_moderate,
    "wcc_strong": wcc_strong
})
