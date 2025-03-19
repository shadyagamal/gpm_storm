#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load trained SOM, assign BMUs, and visualize cluster properties.

@author: shadya
"""
#%% 
import os
import sys
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Import local functions
PACKAGE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm.visualization import plot_cartopy_background # type: ignore
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

#%%
# Configuration
PARALLEL = False
FILEPATH = os.path.expanduser("~/gpm_storm/data/patch_statistics2.parquet")  
SOM_DIR = os.path.expanduser("~/gpm_storm/script")  # Update if needed
FIGS_DIR = os.path.expanduser("~/gpm_storm/figs")  
SOM_NAME = "zonal_SOM"  # Change for different experiments
VARIABLE = "precipRateNearSurface"
NUM_IMAGES = 25
NCOLS = 5

if PARALLEL: 
    create_dask_cluster() 
    
figs_som_dir = os.path.join(FIGS_DIR, SOM_NAME)
os.makedirs(figs_som_dir, exist_ok=True)
    
#%% --------------------------------------------------------------------------------.
# Read the Parquet file into a DataFrame
df = pd.read_parquet(FILEPATH)

# Define the features to train the SOM 
info_dict = get_experiment_info(SOM_NAME)  # HERE INSIDE YOU DEFINE THE EXPERIMENT (features, som_settings ...)
features = info_dict["features"]
n_rows, n_columns = info_dict["som_grid_size"] 

# Subset here the dataframe row to discard (i.e. nan stuffs, select only high intensity ...)
# TODO: TO IMPROVE ! IDEALLY PARAMETRIZE IN get_experiment_info !
# How can you recall the preprocessing ... if you modify manually the code for each trained som !!!! 
# # for feature in features:
#     df = filter_nan_values(df, features)
df = df.dropna(subset=features) # MAYBE THIS IS ENOUGH for the moment ... but are discarding lot of stuffs... to be reported ! 

# Load SOM 
som = load_som(som_dir=SOM_DIR, som_name=SOM_NAME)

#%% Get the Best Matching Units (BMUs) for each data point
bmus = som.bmus

# Add to dataframe 
df['row'] = bmus[:, 0]
df['col'] = bmus[:, 1]

#### Define SOM nodes dataframes
arr_df = create_som_df_array(som=som, df=df)

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
#%% Plot the SOM grid with sample images
arr_ds = create_som_sample_ds_array(arr_df,
                                    variables=VARIABLES,
                                    parallel=PARALLEL)

img_fpath = os.path.join(figs_som_dir, "som_grid_samples.png")
figsize=(10, 10)


fig = plot_som_array_datasets(arr_ds, figsize=figsize, variable=VARIABLE)
fig.tight_layout()
fig.savefig(img_fpath)
fig.close()


#%% Plot SOM node samples 

num_images = 25
ncols=5
figsize=(15, 15)
    

for row in range(n_rows):
    for col in range(n_columns):
        img_fpath = os.path.join(figs_som_dir, f"dir_path_{row}_{col}_nodes_sample.png")
        img_fpath_map = os.path.join(figs_som_dir, f"dir_path_{row}_{col}_nodes_map.png")

        df_node = arr_df[row, col]
        list_ds = sample_node_datasets(df_node, num_images=num_images,
                                       variables=VARIABLE,
                                       parallel=PARALLEL)

        fig = plot_images(list_ds, ncols=ncols, figsize=figsize, variable=VARIABLE)
        fig.tight_layout()
        fig.savefig(img_fpath)
        plt.close(fig)  # Close the figure to release resources
        
        
        df_subset = df[np.logical_and(df["row"] == row, df["col"] == col)]
        # Extract month from the "time" variable
        df_subset["time"] = pd.to_datetime(df_subset["time"])
        df_subset.loc[:, "month"] = df_subset["time"].dt.month  # Use .loc to avoid warning
        
        lon = df_subset["lon"].values
        lat = df_subset["lat"].values
        value = df_subset["precipitation_average"]
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        plot_cartopy_background(ax)
        
        # Use the extracted month to color data points
        sc = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=df_subset["month"], s=2)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label('Month')
        
        fig.savefig(img_fpath_map)
        plt.close(fig)
        #### Plot node feature statistics 
df_stats = create_som_df_features_stats(df)
fig = plot_som_feature_statistics(df_stats, feature='precipitation_average')