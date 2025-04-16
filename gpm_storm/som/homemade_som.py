#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:03:42 2025

@author: gamal
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from gpm_storm.som.som_metrics import topographic_error, quantization_error
from somperf.metrics import *
import time
import os
import random
import glob
import xarray as xr
from matplotlib import colors
import cartopy.crs as ccrs
from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.plot import (
    plot_images,
)
from sklearn.cluster import KMeans
import umap

# Functions
def preprocess_data(df, features):
    df_cleaned = df.copy()
    fill_zero_cols = [
        "P_GT1_mean", "P_GT1_sum",
        "MA_LP_GT_0", "MiA_LP_GT_0", "MA_LP_GT_1", "MiA_LP_GT_1",
        "P_GT2_mean", "P_GT2_sum", "MA_LP_GT_2", "MiA_LP_GT_2",
        "P_GT5_mean", "P_GT5_sum", "MA_LP_GT_5", "MiA_LP_GT_5",
        "P_GT10_mean", "P_GT10_sum", "MA_LP_GT_10", "MiA_LP_GT_10",
        "P_GT20_mean", "P_GT20_sum", "MA_LP_GT_20", "MiA_LP_GT_20",
        "P_GT50_mean", "P_GT50_sum",
        "P_GT80_mean", "P_GT80_sum",
        "P_GT120_mean", "P_GT120_sum",
        "LCC_30_mean", "LCC_30_std", "ICC_30_mean", "ICC_30_std",
        "LCC_40_mean", "LCC_40_std", "ICC_40_mean", "ICC_40_std",
        "LCC_30_max", "ICC_30_max", "LCC_40_max", "ICC_40_max"]
    
    for col in fill_zero_cols:
        if col in df_cleaned.columns:
            df[col] = df_cleaned[col].fillna(0)
    df_cleaned = df.dropna(axis=1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned[df_cleaned.columns[:-9]]),
        columns=df_cleaned.columns[:-9],
        index=df_cleaned.index,
    )
    return df_scaled

def precompute_distances(m, k):
    distance_matrix = np.zeros((m, k, m, k))
    for i1 in range(m):
        for j1 in range(k):
            for i2 in range(m):
                for j2 in range(k):
                    distance_matrix[i1, j1, i2, j2] = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)
    return distance_matrix

def update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma):
    decay = np.exp(-epoch / (n_epochs / 2))
    eta = initial_eta * decay
    sigma = initial_sigma * np.exp(-epoch / (n_epochs * 0.75))
    return eta, sigma

def update_weights(X_batch, original_indices, W, distance_matrix, current_bmus, eta, sigma):
    delta_W = np.zeros_like(W)
    for x, original_idx in zip(X_batch, original_indices):
        dists = np.linalg.norm(W - x, axis=2)
        bmu = np.unravel_index(np.argmin(dists), dists.shape)
        current_bmus[original_idx] = bmu

        distances = distance_matrix[bmu[0], bmu[1]]
        h = np.exp(-distances / (sigma ** 2))[:, :, np.newaxis]
        delta_W += eta * h * (x - W)
    return W + delta_W / X_batch.shape[0]

def compute_bmu_metrics(prev_bmus, current_bmus):
    grid_distances = np.abs(current_bmus - prev_bmus).sum(axis=1)
    mean_movement = np.mean(grid_distances)
    switch_flags = np.any(current_bmus != prev_bmus, axis=1)
    switch_count = np.sum(switch_flags)
    return mean_movement, switch_count, current_bmus.copy()

def quantization_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    distances = cdist(X, flat_W)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)

def topographic_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    codesize = W.shape[:2]
    errors = []

    for x in X:
        dists = np.linalg.norm(flat_W - x, axis=1)
        bmu_indices = np.argsort(dists)[:2]
        bmu1 = np.unravel_index(bmu_indices[0], codesize)
        bmu2 = np.unravel_index(bmu_indices[1], codesize)
        dist = np.linalg.norm(np.array(bmu1) - np.array(bmu2))
        errors.append(0 if dist == 1 else 1)

    return np.mean(errors)

def compute_som_quality_metrics(X, W, current_bmus):
    qe = quantization_error(W, X)
    te = topographic_error(W, X)
    return qe, te


def train_som_with_convergence(X, W, distance_matrix, n_epochs=100, sigma=1.5, eta=0.75, batch_size=64):
    m, k, dim = W.shape
    n_samples = X.shape[0]
    initial_eta = eta
    initial_sigma = sigma

    prev_bmus = np.zeros((n_samples, 2), dtype=int)
    bmu_movement = []
    bmu_switch_count = []
    quant_errors = []
    topo_errors = []

    for epoch in range(n_epochs):
        bmu_counts = np.zeros((m, k), dtype=int)
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        current_bmus = np.zeros((n_samples, 2), dtype=int)
        
        decay = np.exp(-epoch / (n_epochs / 2)) 
        eta = initial_eta * decay
        sigma = initial_sigma * np.exp(-epoch / (n_epochs * 0.75)) 

        for i in range(0, n_samples, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_samples))
            batch = X_shuffled[batch_slice]
            original_indices = shuffled_indices[batch_slice]
            delta_W = np.zeros_like(W)

            for x, original_idx in zip(batch, original_indices):
                dists = np.linalg.norm(W - x, axis=2)
                bmu = np.unravel_index(np.argmin(dists), dists.shape)
                current_bmus[original_idx] = bmu
                bmu_counts[bmu] += 1
        
                distances = distance_matrix[bmu[0], bmu[1]]
                h = np.exp(-distances / (sigma ** 2))[:, :, np.newaxis]
                delta_W += eta * h * (x - W)
            W += delta_W / batch.shape[0]

        grid_distances = np.abs(current_bmus - prev_bmus).sum(axis=1)
        bmu_movement.append(np.mean(grid_distances))
        switch_flags = np.any(current_bmus != prev_bmus, axis=1)
        # switch_rate = np.sum(switch_flags) / n_samples
        bmu_switch_count.append(np.sum(switch_flags))
        prev_bmus = current_bmus.copy()
        
        # if switch_rate < 0.10:
        #     print(f"Stopping early at epoch {epoch}: convergence reached.")
        #     break
        qe = quantization_error(W, X)
        te = topographic_error(W, X)
        quant_errors.append(qe)
        topo_errors.append(te)
        plot_rgb_som(W, bmu_counts)
        

    return W, {"bmu_movement": bmu_movement,
                "bmu_switches": bmu_switch_count,
                "quantization_error": quant_errors,
                "topographic_error": topo_errors}

# def train_som_with_convergence(X, W, distance_matrix, n_epochs=100, min_epochs=25, sigma=1.5, eta=0.5, batch_size=64, epsilon = 0.001):
#     m, k, dim = W.shape
#     n_samples = X.shape[0]
#     initial_eta, initial_sigma = eta, sigma

#     prev_bmus = np.zeros((n_samples, 2), dtype=int)
#     bmu_movement, bmu_switch_count = [], []
#     quant_errors, topo_errors = [], []

#     for epoch in range(n_epochs):
#         plot_rgb_som(W)

#         shuffled_indices = np.random.permutation(n_samples)
#         X_shuffled = X[shuffled_indices]
#         current_bmus = np.zeros((n_samples, 2), dtype=int)

#         eta, sigma = update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma)

#         for i in range(0, n_samples, batch_size):
#             batch_slice = slice(i, min(i + batch_size, n_samples))
#             X_batch = X_shuffled[batch_slice]
#             original_indices = shuffled_indices[batch_slice]
#             W = update_weights(X_batch, original_indices, W, distance_matrix, current_bmus, eta, sigma)

#         movement, switches, prev_bmus = compute_bmu_metrics(prev_bmus, current_bmus)
#         bmu_movement.append(movement)
#         bmu_switch_count.append(switches)

#         qe, te = compute_som_quality_metrics(X, W, current_bmus)
#         quant_errors.append(qe)
#         topo_errors.append(te)
        
#         if epoch >= min_epochs and abs(qe - te) < epsilon:
#             print(f"Stopping early at epoch {epoch} due to QE ≈ TE (|{qe:.4f} - {te:.4f}| < {epsilon})")
#             break

#     return W, {"bmu_movement": bmu_movement,
#                "bmu_switches": bmu_switch_count,
#                 "quantization_error": quant_errors,
#                 "topographic_error": topo_errors}

def find_bmu(W, x):
    dist = np.linalg.norm(W - x, axis=2)
    return np.unravel_index(np.argmin(dist), dist.shape)

def compute_u_matrix(W):
    m, k, _ = W.shape
    u_matrix = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            neighbors = [
                W[ni, nj]
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                if (di != 0 or dj != 0)
                and 0 <= (ni := i + di) < m
                and 0 <= (nj := j + dj) < k
            ]
            u_matrix[i, j] = np.mean([np.linalg.norm(W[i, j] - n) for n in neighbors])
    return u_matrix

def map_samples_to_nodes(X, W):
    return [find_bmu(W, x) for x in X]

def plot_u_matrix(u_matrix):
    plt.figure(figsize=(8, 6))
    plt.title("U-Matrix")
    plt.imshow(u_matrix, cmap="bone", interpolation="nearest")
    plt.colorbar(label='Average Distance')
    plt.show()

def plot_samples(X, bmus, labels=None, label_names=None, colors=None):
    plt.figure(figsize=(8, 6))
    plt.title("Samples Mapped to SOM Grid")
    for i, (y, x) in enumerate(bmus):
        kwargs = {"edgecolors": 'k', "linewidths": 0.3, "alpha": 0.6}
        if labels is not None:
            plt.scatter(x + 0.5, y + 0.5, color=colors[labels[i]], label=label_names[labels[i]], **kwargs)
        else:
            plt.scatter(x + 0.5, y + 0.5, color=X[i], s=100, **kwargs)

    if labels is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_names[i],
                              markerfacecolor=colors[i], markersize=10, markeredgecolor='k')
                   for i in range(len(label_names))]
        plt.legend(handles=handles, title="Classes")

    plt.xlim(0, W.shape[1])
    plt.ylim(0, W.shape[0])
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

def plot_rgb_som(W, bmu_counts=None):
    image = W.copy()
    plt.figure(figsize=(8, 6))
    plt.title("SOM RGB Map")
    plt.imshow(image, interpolation="nearest")
    plt.gca().invert_yaxis()
    plt.grid(False)
    if bmu_counts is not None:
       for i in range(m):
           for j in range(k):
               count = bmu_counts[i, j]
               if count > 0:
                   plt.text(j, i, str(count), ha='center', va='center', fontsize=8,
                            color='white' if np.mean(W[i, j]) < 0.5 else 'black',
                            bbox=dict(facecolor='black', alpha=0.3, lw=0))
    plt.show()
    
def generate_skewed_rgb_dataset(n_samples=1000, red_ratio=0.7):
    """
    Generate an RGB dataset with red overrepresented.
    """
    n_red = int(n_samples * red_ratio)
    n_other = n_samples - n_red

    # Red-dominant samples (bright reds)
    red_samples = np.random.uniform(low=[0.8, 0.0, 0.0], high=[1.0, 0.2, 0.2], size=(n_red, 3))
    
    # Other colors (uniformly spread)
    other_samples = np.random.uniform(0, 1, size=(n_other, 3))

    X = np.vstack([red_samples, other_samples])
    np.random.shuffle(X)
    return X

# --- RGB Dataset ---
N = 1000
X_rgb_scaled = np.random.rand(N,3)
X_skewed = generate_skewed_rgb_dataset(n_samples=2000, red_ratio=0.8)

# UMAP
# reducer = umap.UMAP()

# joblib.dump(reducer, "umap_model.joblib")
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_skewed)

n_clusters = 9
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans.labels_, cmap="tab10", alpha=0.5)
plt.colorbar(label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP with K-Means Clustering")
plt.grid(True)
plt.show()



m, k = 3, 3
n_clusters = m * k
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
labels = kmeans.labels_

# Compute RGB mean for each cluster
W_init = np.zeros((m, k, 3))
for i in range(n_clusters):
    cluster_rgb_values = X_skewed[labels == i]
    if len(cluster_rgb_values) > 0:
        mean_rgb = cluster_rgb_values.mean(axis=0)
    else:
        mean_rgb = np.random.rand(3)  # Fallback if a cluster ends up empty
    W_init[i // k, i % k] = mean_rgb
    
plot_rgb_som(W_init)

m, k = 3, 3
distance_matrix = precompute_distances(m, k)
som_shape = (m, k)  
W = np.random.rand(m, k, X_skewed.shape[1])
G = nx.grid_2d_graph(m, k)
plot_rgb_som(W)
start_time = time.time()
# W_trained = train_som(X_rgb_scaled, W, G)
# W_trained = train_som_optimized(X_rgb_scaled, W, distance_matrix)
# W_trained, metrics = train_som_epochwise(X_rgb_scaled, W, distance_matrix)
W_trained, metrics = train_som_with_convergence(X_skewed, W, distance_matrix, n_epochs=10)
# W_trained, metrics = train_som_with_convergence(X_rgb_scaled, W, distance_matrix, n_epochs=100)
end_time = time.time()
total_time = end_time-start_time
print(total_time)
plot_rgb_som(W_trained)

# bmu_locations = map_samples_to_nodes(X_rgb_scaled, W_trained)
# plot_samples(X_rgb_scaled, bmu_locations)


plt.plot(metrics["bmu_movement"], label="BMU Movement Distance")
plt.plot(metrics["bmu_switches"], label="BMU Switch Count")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("SOM Convergence Metrics")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(metrics['quantization_error'], label='QE')
plt.plot(metrics['topographic_error'], label='TE')
plt.legend()
plt.title("SOM Quality Metrics")
plt.grid(True)
plt.show()


# # --- GPM STORM Dataset ---

# filepath = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
# df = pd.read_parquet(filepath) 
# som_dir = os.path.expanduser("~/gpm_storm/scripts")  
# vars = df.columns[0:-9]
# df_scaled = preprocess_data(df, vars)
# df_sample = df_scaled.sample(n=100000, random_state=42)
# X = df_sample.to_numpy()

# m, k = 10, 10
# dim = X.shape[1]
# W = np.random.rand(m, k, dim)
# distance_matrix = precompute_distances(m, k)
# G = nx.grid_2d_graph(m, k)

# st = time.time()
# W_trained, metrics = train_som_with_convergence(X, W, distance_matrix, n_epochs=100)
# et = time.time()
# tt = et-st
# print(tt)

# plt.plot(metrics["bmu_movement"], label="BMU Movement Distance")
# plt.plot(metrics["bmu_switches"], label="BMU Switch Count")
# plt.xlabel("Epoch")
# plt.ylabel("Metric Value")
# plt.title("SOM Convergence Metrics")
# plt.legend()
# plt.grid(True)
# plt.show()




# # --- VISUAL ----
# VARIABLE = "precipRateNearSurface"
# HEATMAP_VARIABLE = "P_mean"
# SOM_SHAPE = (10, 10)
# NUM_IMAGES = 25
# zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr" 

# def get_patch_dataset(granule_id, patch_id, time, cache):
#     year, month = time.year, time.month
#     if granule_id in cache:
#         ds = cache[granule_id]
#     else:
#         search_path = os.path.join(zarr_directory, f"{year:04d}/{month:02d}", "*.zarr")
#         zarr_files = glob.glob(search_path)
#         for zarr_file in zarr_files:
#             if granule_id in os.path.basename(zarr_file):
#                 ds = xr.open_zarr(zarr_file)
#                 cache[granule_id] = ds
#                 break
#         else:
#             return None
#     return ds.isel(patch=patch_id)


# def create_node_image_array(weights, df):
#     n_rows, n_cols = weights.shape[:2]
#     arr_df = np.empty((n_rows, n_cols), dtype=object)
#     arr_ds = np.empty((n_rows, n_cols), dtype=object)

#     for r in range(n_rows):
#         for c in range(n_cols):
#             mask = (df["row"] == r) & (df["col"] == c)
#             arr_df[r, c] = df[mask].reset_index(drop=True)
#             arr_ds[r, c] = None  # We'll populate this later if needed
#     return arr_df, arr_ds



# def plot_som_grid(arr_ds, cmap="turbo", norm=None):
#     n_rows, n_cols = arr_ds.shape
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

#     for r in range(n_rows):
#         for c in range(n_cols):
#             ax = axes[r, c]
#             ax.axis("off")
#             ds = arr_ds[r, c]
#             if ds is not None:
#                 da = ds["precipRateNearSurface"]
#                 da.plot.imshow(ax=ax, cmap=cmap, norm=norm,
#                                add_colorbar=False, add_labels=False)

#     plt.tight_layout()
#     plt.show()



# def plot_node_samples_and_maps(arr_df, df, zarr_directory, figs_som_dir, variable="precipRateNearSurface", Ncols=5, num_images=25):
#     n_rows, n_cols = arr_df.shape
#     for r in range(n_rows):
#         for c in range(n_cols):
#             df_node = arr_df[r, c]
#             if df_node.empty:
#                 continue

#             img_fpath = os.path.join(figs_som_dir, f"node_{r}_{c}_samples.png")
#             img_fpath_map = os.path.join(figs_som_dir, f"node_{r}_{c}_map.png")

#             try:
#                 random_indices = random.sample(range(len(df_node)), min(num_images, len(df_node)))
#                 list_ds = []

#                 for index in random_indices:
#                     patch_row = df_node.iloc[index]
#                     granule_id = str(patch_row["gpm_granule_id"])
#                     patch_id = patch_row["patch_id"]
#                     time = pd.to_datetime(patch_row["time"])
#                     year, month = time.year, time.month

#                     search_path = os.path.join(zarr_directory, f"{year:04d}/{month:02d}", "*.zarr")
#                     zarr_files = glob.glob(search_path)

#                     for zarr_file in zarr_files:
#                         if granule_id in os.path.basename(zarr_file):
#                             ds = xr.open_zarr(zarr_file)
#                             list_ds.append(ds.isel(patch=patch_id))
#                             break

#                 fig = plot_images(list_ds, ncols=Ncols, figsize=(15, 15), variable=variable)
#                 fig.tight_layout()
#                 fig.savefig(img_fpath)
#                 plt.close(fig)

#                 df_subset = df_node.copy()
#                 df_subset["time"] = pd.to_datetime(df_subset["time"])
#                 df_subset["month"] = df_subset["time"].dt.month
#                 lon, lat = df_subset["lon"].values, df_subset["lat"].values

#                 fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
#                 plot_cartopy_background(ax)
#                 sc = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=df_subset["month"], s=2)
#                 cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
#                 cbar.set_label("Month")
#                 fig.savefig(img_fpath_map)
#                 plt.close(fig)

#             except Exception as e:
#                 print(f"⚠️ Error in node ({r},{c}): {e}")


# def plot_mean_heatmap(arr_df, variable="P_mean", cmap="viridis"):
#     n_rows, n_cols = arr_df.shape
#     mean_values = np.full((n_rows, n_cols), np.nan)

#     for r in range(n_rows):
#         for c in range(n_cols):
#             df_node = arr_df[r, c]
#             if not df_node.empty:
#                 mean_values[r, c] = df_node[variable].mean()

#     masked = np.ma.masked_invalid(mean_values)
#     plt.figure(figsize=(8, 8))
#     plt.imshow(masked, cmap=cmap, origin="upper")
#     cbar = plt.colorbar()
#     cbar.set_label(f"Mean {variable}")
#     plt.title(f"Mean {variable} per SOM Node")
#     plt.xlabel("SOM Column")
#     plt.ylabel("SOM Row")
#     plt.xticks(np.arange(n_cols))
#     plt.yticks(np.arange(n_rows))
#     plt.grid(False)
#     plt.show()

    
# def compute_bmus(data, weights):
#     # data: (n_samples, n_features)
#     # weights: (n_rows, n_cols, n_features)
#     n_rows, n_cols = weights.shape[:2]
#     flat_weights = weights.reshape(n_rows * n_cols, -1)  # (n_nodes, n_features)

#     # Compute Euclidean distance to each SOM node
#     dists = np.linalg.norm(data[:, None, :] - flat_weights[None, :, :], axis=2)  # (n_samples, n_nodes)

#     bmu_indices = np.argmin(dists, axis=1)  # (n_samples,)
#     bmu_rows, bmu_cols = np.divmod(bmu_indices, n_cols)
#     return np.stack([bmu_rows, bmu_cols], axis=1)  # (n_samples, 2)

# def update_dataframe_with_bmus(df, features, weights):
#     bmus = compute_bmus(features, weights)
#     df["row"] = bmus[:, 0]
#     df["col"] = bmus[:, 1]
#     return df


# df_sample = update_dataframe_with_bmus(df_sample, X, W_trained)

# arr_df, arr_ds = create_node_image_array(W_trained, df_sample)
# plot_som_grid(arr_ds, cmap="turbo", norm=colors.LogNorm(vmin=0.01, vmax=300))
# plot_node_samples_and_maps(arr_df, df_sample, zarr_directory, figs_som_dir)
# plot_mean_heatmap(arr_df, variable="P_mean")
