import itertools
import os
import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colors, cm
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
)
import seaborn as sns


filepath = "/home/gamal/gpm_storm/data/merged_data_total_0_with_bmus.parquet"
df_bmu = pd.read_parquet(filepath)

simple_filepath = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
simple_df = pd.read_parquet(simple_filepath)
vars = simple_df.columns[0:-9]
simple_df = simple_df.dropna(subset=vars)

total_filepath = "/home/gamal/gpm_storm/data/merged_data_total_0_with_bmus_umap_kmeans.parquet"
df = pd.read_parquet(total_filepath)

# df_subset = df.dropna(subset=vars)
# df_subset = df_subset.reset_index(drop=True)
# df_bmu = df_bmu.reset_index(drop=True)
# df_subset["row"] = df_bmu["row"]
# df_subset["col"] = df_bmu["col"]
# df["row"] = np.nan
# df["col"] = np.nan
# df.loc[df_subset.index, ["row", "col"]] = df_subset[["row", "col"]]
# new_filepath = os.path.expanduser("~/gpm_storm/data/merged_data_total_0_with_bmus_umap_kmeans.parquet")
# df.to_parquet(new_filepath)


for col in df_bmu.columns:
    print(col)
    
grouped = df_bmu.groupby(['row', 'col'])


summary = grouped.mean(numeric_only=True)
summary_std = grouped.std(numeric_only=True)
counts = grouped.size().unstack(fill_value=0)


mean_reflect = grouped['lat'].mean().unstack()
sns.heatmap(mean_reflect, cmap="viridis", annot=False)
plt.title("Mean Max Reflectivity per SOM node")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()


# Per node
node_df = df_bmu[(df_bmu['row'] == 1) & (df_bmu['col'] == 0)]
print(f"{len(node_df)} events in node (3, 9)")


features = ["P_mean", "P_max", "P_count", "REFC_mean", "REFCH_mean", "CC_30_count", "lon", "lat"]
for var in features:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(df_bmu[var], label="All data", linewidth=2)
    sns.kdeplot(node_df[var], label="Node (12,5)", linewidth=2)
    plt.title(f"Distribution of {var}")
    plt.legend()
    plt.tight_layout()
    plt.show()


node_corr = node_df.corr(numeric_only=True)
sns.heatmap(node_corr, cmap="coolwarm", center=0)
plt.title("Correlations inside node (12,5)")
plt.show()


plt.figure(figsize=(6, 4))
plt.scatter(df_bmu["lon"], df_bmu["lat"], s=5, alpha=0.1, label="All")
plt.scatter(node_df["lon"], node_df["lat"], s=10, alpha=0.9, label="Node (12,5)")
plt.title("Geographical distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Cluster Composition Analysis
# For each SOM node, what are the dominant UMAP+KMeans clusters?
# For each KMeans cluster, what are the dominant SOM nodes?

cross_table = pd.crosstab(index=[df["row"], df["col"]], columns=df["kmeans_cluster"])
sns.heatmap(cross_table, cmap="mako")
plt.title("SOM node vs UMAP+KMeans cluster membership")
plt.show()


# Summary Statistics per KMeans Cluster
grouped_kmeans = df.groupby('kmeans_cluster')
summary_kmeans = grouped_kmeans.mean(numeric_only=True)
summary_kmeans_std = grouped_kmeans.std(numeric_only=True)

# Feature Distributions per Cluster
for var in features:
    plt.figure(figsize=(6, 3))
    sns.kdeplot(df[var], label="All data", linewidth=2)
    sns.kdeplot(df[df['kmeans_cluster'] == 5][var], label="Cluster 5", linewidth=2)
    plt.title(f"Distribution of {var}")
    plt.legend()
    plt.tight_layout()
    plt.show()

n_rows, n_cols = 10, 10

for var in df.columns[:-16]:
    # Create a 10x10 grid of subplots for each variable with shared axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    
    # Loop through each cluster and plot its distribution
    for i, cluster in enumerate(sorted(df['kmeans_cluster'].unique())):
        ax = axes[i]  # Get the axis for the current cluster
        
        # Plot the distribution of the variable for the current cluster
        sns.kdeplot(df[df['kmeans_cluster'] == cluster][var], label=f"Cluster {cluster}", linewidth=2, ax=ax)
        
        # Set title and labels for each subplot
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        
    # Adjust layout for each variable's subplot grid
    plt.tight_layout()
    plt.suptitle(f"Distributions of {var} by Cluster", fontsize=16)
    plt.subplots_adjust(top=0.92)  # Adjust title placement
    plt.show()


# Geographical Footprint per Cluster
plt.figure(figsize=(6, 4))
plt.scatter(df["lon"], df["lat"], s=5, alpha=0.1, label="All")
plt.scatter(df[df["kmeans_cluster"] == 5]["lon"], df[df["kmeans_cluster"] == 5]["lat"], s=10, alpha=0.9, label="Cluster 5")
plt.title("Geographical distribution of Cluster 5")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

