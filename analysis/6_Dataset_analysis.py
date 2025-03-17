#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:29:26 2023

@author: comi
"""

import glob
import os
import pandas as pd 
import numpy as np
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, shapiro, anderson
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



#%%

def _relative_distribution_of_dataset(df):

    columns = df.columns

    # Calculate the number of rows and columns needed for the subplots
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed

    # Create subplots with the desired layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle("Distributions of Data", y=1.02)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot the relative distributions for each column
    for i, column in enumerate(columns):
        # Check if the column is numeric (excluding the 'time' column)
        if pd.api.types.is_numeric_dtype(df[column]) and column != ('time', 'along_track_start', 'along_track_stop', 'gpm_granule_id'):
            sns.histplot(data=df, x=column, kde=True, ax=axes[i], stat='percent', common_norm=False)
            axes[i].set_title(f"Relative Distribution of {column}")
            axes[i].set_xlabel("Values")
            axes[i].set_ylabel("Relative Frequency (%)")

    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    
def _relative_log_distribution_of_dataset(df):
    columns = df.columns

    # Calculate the number of rows and columns needed for the subplots
    num_columns = 30
    num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed

    # Create subplots with the desired layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle("Distributions of Data", y=1.02)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot the relative distributions for each column
    for i, column in enumerate(columns[:30]):  # Plot only the first twenty variables
        # Check if the column is numeric (excluding the 'time' column)
        if pd.api.types.is_numeric_dtype(df[column]) and column != ('time', 'along_track_start', 'along_track_stop', 'gpm_granule_id'):
            # Use log scale on the x-axis
            sns.histplot(data=df, x=column, kde=True, ax=axes[i], stat='percent', common_norm=False, log_scale=(True, False))
            axes[i].set_title(f"Relative Distribution of {column}")
            axes[i].set_xlabel("Values (log scale)")
            axes[i].set_ylabel("Relative Frequency (%)")

    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def _boxplot_of_dataset(df):
        # Get the list of column names in your DataFrame
    columns = df.columns
    
    # Calculate the number of rows and columns needed for the subplots
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Adjust the number of columns as needed
    
    # Create subplots with the desired layout
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    fig.suptitle("Boxplots of Data", y=1.02)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Plot boxplots for each column
    for i, column in enumerate(columns):
        # Check if the column is numeric (excluding the 'time' column)
        if pd.api.types.is_numeric_dtype(df[column]) and column != ('time', 'along_track_start', 'along_track_stop', 'gpm_granule_id'):
            sns.boxplot(x=column, data=df, ax=axes[i], showfliers=False)
            axes[i].set_title(f"Boxplot of {column}")
            axes[i].set_xlabel("Values")
    
    # Hide any empty subplots
    for i in range(num_columns, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()
    
def _bivariate_analysis(df, x_variable, y_variable, color_variable):
    
    
    df[x_variable] = pd.to_numeric(df[x_variable], errors='coerce')
    df[y_variable] = pd.to_numeric(df[y_variable], errors='coerce')
    df[color_variable] = pd.to_numeric(df[color_variable], errors='coerce')

    # Convert Arrow columns to Pandas Series
    x_values = np.array(df[x_variable])
    y_values = np.array(df[y_variable])
    color_values = np.array(df[color_variable])
    
    # Handle missing values
    df.dropna(subset=[x_variable, y_variable, color_variable], inplace=True)



    # Set the style of seaborn
    sns.set(style="whitegrid")
    
    # Create a scatter plot with color based on the third variable
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_values, y=y_values, hue=color_values, palette='viridis', edgecolor='w', s=100)
    
    # Add labels and a legend
    plt.title(f'Bivariate Analysis of {x_variable} and {y_variable} (Colored by {color_variable})')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.legend(title=color_variable)
    
    # Show the plot
    plt.show()
    
    bins = 20
    
    # Create a 2D histogram with mean values
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(
        x=x_values,
        y=y_values,
        values=color_values,
        statistic='mean',
        bins=bins
    )
    
    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(statistic.T, cmap='viridis', xticklabels=x_edges, cbar=True)
    
 
    # Add labels and a title
    plt.title(f'Bivariate Analysis with Mean Values (Color Coded for {color_variable})')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    
    # Show the plot
    plt.show()

def _normality_tests(df):    
   columns_to_process = df.columns[:-5]
   results_list = []

   for column in columns_to_process:
       # Normality Test
       stat_norm, p_value_norm = shapiro(df[column])
       
       # Log-Normality Test
       log_data = np.log(df[column])
       stat_log_norm, p_value_log_norm = shapiro(log_data)
       
       # Anderson-Darling Test
       result_anderson = anderson(df[column])
       
       results_list.append({
           'Variable': column,
           'Shapiro-Wilk Statistic': stat_norm,
           'Shapiro-Wilk p-value': p_value_norm,
           'Log-Normal Shapiro-Wilk Statistic': stat_log_norm,
           'Log-Normal Shapiro-Wilk p-value': p_value_log_norm,
           'Anderson-Darling Statistic': result_anderson.statistic,
           'Anderson-Darling Critical Values': result_anderson.critical_values,
           'Anderson-Darling Significance Level': result_anderson.significance_level
       })

   results_df = pd.DataFrame(results_list)
  
    # results_df["Variable"]: column
    # results_df['Shapiro-Wilk Statistic']: stat_norm
    # results_df['Shapiro-Wilk p-value']: p_value_norm
    # results_df['Log-Normal Shapiro-Wilk Statistic']: stat_log_norm
    # results_df['Log-Normal Shapiro-Wilk p-value']: p_value_log_norm
    # results_df['Anderson-Darling Statistic']: result_anderson.statistic
    # results_df['Anderson-Darling Critical Values']: result_anderson.critical_values
    # results_df['Anderson-Darling Significance Level']: result_anderson.significance_level
      
   return results_df

def _process_nan_values(df, threshold_percentage=30):
    # Calculate the percentage of NaN values for each column
    nan_percentages = (df.isnull().sum() / len(df)) * 100

    # Display NaN percentages
    print("NaN Percentages for Each Column:")
    print(nan_percentages)

    # Create a list of columns to drop based on the threshold percentage
    columns_to_drop = nan_percentages[nan_percentages > threshold_percentage].index.tolist()

    # Create a new DataFrame without columns containing NaN values above the threshold
    df_no_nan = df.drop(columns=columns_to_drop)

    return df_no_nan


def filter_nan_values(df, variable_name):
    """ Filter out rows where a specified variable has NaN values."""
    filtered_df = df.dropna(subset=[variable_name])
    return filtered_df

def spacial_analysis(df, color_variable, lat_variable="lat", lon_variable="lon"):
    # Convert latitude, longitude, and color variables to numeric
    df[lat_variable] = pd.to_numeric(df[lat_variable], errors='coerce')
    df[lon_variable] = pd.to_numeric(df[lon_variable], errors='coerce')
    
    df[color_variable] = pd.to_numeric(df[color_variable], errors='coerce')

    # Drop rows with missing values in specified columns
    df = df.dropna(subset=[lat_variable, lon_variable, color_variable])

    # Create scatter plot with hexbin for binned heatmap and average color
    plt.figure(figsize=(12, 10))
    hb = plt.hexbin(
        df[lon_variable],
        df[lat_variable],
        C=df[color_variable],
        gridsize=20,  # Adjust gridsize as needed
        cmap='viridis',
        edgecolor='w',
        reduce_C_function=np.mean,  # Calculate mean value in each bin
        mincnt=1  # Minimum number of points in a bin to be colored
    )

    # Add labels and a title
    plt.title(f'Spatial Analysis with Average {color_variable} in Bins')
    plt.xlabel(lon_variable)
    plt.ylabel(lat_variable)

    # Add a colorbar
    cbar = plt.colorbar(hb, label=color_variable)

    # Show the plot
    plt.show()

import cartopy.crs as ccrs
from gpm.visualization.plot import plot_cartopy_background, plot_colorbar
from gpm.bucket.analysis import pl_add_geographic_bins, pl_df_to_xarray

df_rounded = df.copy() 
df_rounded["lon_bin"] = df_rounded["lon"].round(1)
df_rounded["lat_bin"] = df_rounded["lat"].round(1)
grouped_df = df_rounded.groupby(["lon_bin", "lat_bin"])
binned_df = grouped_df.agg(["count", "median"])

df.dtypes
import polars 
import polars as pl
xbin_column="lon_bin"
ybin_column="lat_bin"
bin_spacing=0.1
bin_spacing=2

df["row-col"] = df["col"].astype(str) + "-" + df["row"].astype(str)
df_pl = polars.from_pandas(df)
df_pl = pl_add_geographic_bins(df_pl, xbin_column=xbin_column, ybin_column=ybin_column, 
                               bin_spacing=bin_spacing, x_column="lon", y_column="lat")

grouped_df = df_pl.groupby([xbin_column, ybin_column])
df_stats_pl = grouped_df.agg(pl.col("precipitation_average").count().alias("bin_count"),
                             pl.col("row-col").mode().alias("more_frequent_node")
                             )

ds = pl_df_to_xarray(df_stats_pl,  
                     xbin_column=xbin_column, 
                     ybin_column=ybin_column, 
                     bin_spacing=bin_spacing)


df_subset = df[np.logical_and(df["row"] == 0, df["col"] == 8)]
lon = df_subset["lon"].values
lat = df_subset["lat"].values
value = df_subset["echodepth30_mean"]

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c="orange", s=2)

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=value, s=4, cmap="Spectral", vmax=5000)
plot_colorbar(p=p, ax=ax)


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ds["bin_count"].plot.imshow(ax=ax, x="longitude", y="latitude", cmap="Spectral", add_colorbar=False)
plot_colorbar(p=p, ax=ax)


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
plot_cartopy_background(ax)
p = ds["row-col"].plot.imshow(ax=ax, x="longitude", y="latitude", cmap="Spectral", add_colorbar=False)
plot_colorbar(p=p, ax=ax)




def preliminary_dataset_analysis(dst_dir):
    list_files = glob.glob(os.path.join(dst_dir, "*", "*", "*", "*.parquet"))
    dataset = ds.dataset(list_files)
    
    table = dataset.to_table()

        
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    
    
    #creating dataset without nan values
    df_no_nan = _process_nan_values(df, threshold_percentage=1)
    
        
    # Get the list of column names in your DataFrame
    
    _relative_distribution_of_dataset(df)
    _boxplot_of_dataset(df)
    _bivariate_analysis(df, 'precipitation_pixel', 'precipitation_sum', 'lat')
    _relative_log_distribution_of_dataset(df)
    results = _normality_tests(df_no_nan)
    
    #scale data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    
    return df, df_no_nan, df_scaled, results 

# dst_dir = "/ltenas8/data/GPM_STORM/features_v1"
#lit_files = preliminary_dataset_analysis(dst_dir)

file_path = "~/gpm_storm/data/patch_statistics.parquet"

# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)
df.columns
  

spacial_analysis(df, color_variable = "precipitation_average")


df_no_nan_variable = filter_nan_values(df, variable_name="aspect_ratio_largest_patch_over_{threshold}")

 
df["lenght_track"] = df["along_track_end"] - df["along_track_start"]
# %%
