#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:31:51 2023

@author: ghiggi
"""
import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from gpm.visualization.plot import plot_cartopy_background
import matplotlib.pyplot as plt
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

parallel = True 
file_path = '/home/comi/Projects/dataframe2.parquet'
som_dir = "/home/comi/Projects/gpm_storm/scripts/" # TODO to change ... 
figs_dir = "/home/comi/Projects/gpm_storm/figs/"

som_name = "zonal_SOM" # TODO: THIS IS THE NAME IDENTIFYING THE EXPERIMENT

variables = "precipRateNearSurface"

if parallel: 
    create_dask_cluster() 
    
figs_som_dir = os.path.join(figs_dir, som_name)
os.makedirs(figs_som_dir, exist_ok=True)
    
#--------------------------------------------------------------------------------.
# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Define the features to train the SOM 
info_dict = get_experiment_info(som_name)  # HERE INSIDE YOU DEFINE THE EXPERIMENT (features, som_settings ...)
features = info_dict["features"]
n_rows, n_columns = info_dict["som_grid_size"] 

# Subset here the dataframe row to discard (i.e. nan stuffs, select only high intensity ...)
# TODO: TO IMPROVE ! IDEALLY PARAMETRIZE IN get_experiment_info !
# How can you recall the preprocessing ... if you modify manually the code for each trained som !!!! 
# # for feature in features:
#     df = filter_nan_values(df, features)
df = df.dropna(subset=features) # MAYBE THIS IS ENOUGH for the moment ... but are discarding lot of stuffs... to be reported ! 

# Load SOM 
som = load_som(som_dir=som_dir, som_name=som_name)

# Get the Best Matching Units (BMUs) for each data point
bmus = som.bmus

# Add to dataframe 
df['row'] = bmus[:, 0]
df['col'] = bmus[:, 1]

#### Define SOM nodes dataframes
arr_df = create_som_df_array(som=som, df=df)


### Plot the SOM grid with sample images
arr_ds = create_som_sample_ds_array(arr_df,
                                    variables=variables,
                                    parallel=parallel)

img_fpath = os.path.join(figs_som_dir, "som_grid_samples.png")
figsize=(10, 10)
variable = "precipRateNearSurface"

fig = plot_som_array_datasets(arr_ds, figsize=figsize, variable=variable)
fig.tight_layout()
fig.savefig(img_fpath)
fig.close()


### Plot SOM node samples 
variable = "precipRateNearSurface"
num_images = 25
ncols=5
figsize=(15, 15)
    

for row in range(n_rows):
    for col in range(n_columns):
        img_fpath = os.path.join(figs_som_dir, f"dir_path_{row}_{col}_nodes_sample.png")
        img_fpath_map = os.path.join(figs_som_dir, f"dir_path_{row}_{col}_nodes_map.png")

        df_node = arr_df[row, col]
        list_ds = sample_node_datasets(df_node, num_images=num_images,
                                       variables=variable,
                                       parallel=parallel)

        fig = plot_images(list_ds, ncols=ncols, figsize=figsize, variable=variable)
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