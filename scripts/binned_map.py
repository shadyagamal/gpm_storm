#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:05:33 2025

@author: gamal
"""
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.som.experiments import save_som
import itertools
import numpy as np
from collections import Counter
from scipy.stats import skew
from somperf.metrics import *
import umap
from sklearn.cluster import KMeans
import pycolorbar
from pycolorbar import get_plot_kwargs  # noqa
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

def get_som_colormap(varname: str) -> str:
    """
    Returns the appropriate colormap for a given SOM variable name.
    """

    # Precipitation-related variables
    if varname.startswith("P_") or varname.startswith("MP_"):
        return "rain_r"

    # Morphology variables
    elif varname.startswith(("MA_", "MiA_", "AR_")):
        return "dense_r"

    # Reflectivity variables
    elif varname.startswith(("REFC_", "REFCH_", "ED", "ETH", "LCC_", "ICC_", "CC_")):
        return "eclipse"

    # Temperature (if present — placeholder, extend as needed)
    elif varname.lower().startswith("temp") or varname.lower().endswith("_temp"):
        return "sunset"

    # Land-type or catch-all (can refine as you grow the dataset)
    else:
        return "dense_r"
    



figs_dir = os.path.expanduser(f"~/gpm_storm/figs/Binned_maps")
filepath0 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
filepath1 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet"
df0 = pd.read_parquet(filepath0) 
df1 = pd.read_parquet(filepath1) 
df = pd.concat([df0,df1], ignore_index=True)

num_df = df.select_dtypes(include='number')
variables =  num_df.columns
colormaps = pycolorbar.colormaps
colorbars = pycolorbar.colorbars

# Binned maps
degree = 1
lat_bins = np.arange(-90, 91, degree)  
lon_bins = np.arange(-180, 181, degree)  
df["lat_bin"] = np.digitize(df["lat"], bins=lat_bins) - 1
df["lon_bin"] = np.digitize(df["lon"], bins=lon_bins) - 1
agg_df = df.groupby(["lat_bin", "lon_bin"]).mean(numeric_only=True).reset_index()
agg_df["lat"] = lat_bins[agg_df["lat_bin"]]
agg_df["lon"] = lon_bins[agg_df["lon_bin"]]

for var in variables:
    color = get_som_colormap(var)
    cmap = colormaps.get_cmap(color)
    heatmap = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)  # Initialize with NaNs
    for _, row in agg_df.iterrows():
        lat_idx = np.searchsorted(lat_bins, row["lat"], side="right") - 1
        lon_idx = np.searchsorted(lon_bins, row["lon"], side="right") - 1
        if 0 <= lat_idx < heatmap.shape[0] and 0 <= lon_idx < heatmap.shape[1]:
            heatmap[lat_idx, lon_idx] = row[var]

    fig, ax = plt.subplots(figsize=(13, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    mesh = ax.pcolormesh(lon_bins, lat_bins, heatmap, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=1)
    cbar.set_label(f"{var}")
    ax.set_xticks(np.arange(-180, 181, 45), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 15), crs=ccrs.PlateCarree())
    ax.tick_params(labelsize=10)
    plt.title(f"Global Distribution of {var} ({degree}°x{degree}° bins)")
    
    save_path = os.path.join(figs_dir, f"Binned_map_{var}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
 

