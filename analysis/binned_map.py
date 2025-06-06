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

# Define color transitions (RGBA or named)
colors = [
    (1.0, 1.0, 1.0),   # white
    (0.29, 0.4, 0.7),   # blue
    (0.25, 0.5, 0.76),   # blue
    (0.01, 0.84, 0.93),   # blue
    (0.77, 0.84, 0.05),   # yellow
    (1.0, 0.92, 0.0),   # yellow
    (1.0, 0.44, 0.0),    # red
    (1.0, 0.0, 0.0),    # red
    (0.56, 0.0, 0.0)    # red
]

# Create the custom colormap
cmap = LinearSegmentedColormap.from_list("white_blue_yellow_red", colors, N=256)


# # Visualize the colormap
# gradient = np.linspace(0, 1, 256).reshape(1, -1)
# plt.imshow(gradient, aspect='auto', cmap=cmap)
# plt.axis("off")
# plt.show()

flag_path = os.path.join(res_dir, "houze_flags.parquet")
houze_df = pd.read_parquet(flag_path)

df_bmu = pd.concat([df_bmu, houze_df], axis=1)
houze_categories = [
    "ISE", "DCC_moderate", "DCC_strong",
    "BSR_moderate", "BSR_strong", "hail_flag"
]


# Binned maps
lat_bins = np.arange(-90, 91, 2)  
lon_bins = np.arange(-180, 181, 2)  
df_bmu["lat_bin"] = np.digitize(df_bmu["lat"], bins=lat_bins) - 1
df_bmu["lon_bin"] = np.digitize(df_bmu["lon"], bins=lon_bins) - 1
agg_df = df_bmu.groupby(["lat_bin", "lon_bin"]).mean(numeric_only=True).reset_index()
agg_df["lat"] = lat_bins[agg_df["lat_bin"]]
agg_df["lon"] = lon_bins[agg_df["lon_bin"]]

for var in houze_categories:
    heatmap = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan) 
    for _, row in agg_df.iterrows():
        lat_idx = np.searchsorted(lat_bins, row["lat"], side="right") - 1
        lon_idx = np.searchsorted(lon_bins, row["lon"], side="right") - 1
        if 0 <= lat_idx < heatmap.shape[0] and 0 <= lon_idx < heatmap.shape[1]:
            heatmap[lat_idx, lon_idx] = row[var]

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    mesh = ax.pcolormesh(lon_bins, lat_bins, heatmap, cmap=cmap, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical")
    cbar.set_label(f"{var} ")
    plt.title(f"Global Distribution of {var} (2°x2° bins)")
    plt.show()
    
    
