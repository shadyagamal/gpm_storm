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
import somoclu



# =========================
# SOM Utilities
# =========================

def precompute_distances(m, k):
    i1, j1, i2, j2 = np.indices((m, k, m, k))
    return np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)

def update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma):
    decay = np.exp(-epoch / (n_epochs / 2))
    eta = initial_eta * decay
    sigma = initial_sigma * np.exp(-epoch / (n_epochs * 0.75))
    return eta, sigma

def find_bmu(W, x):
    dists = np.linalg.norm(W - x, axis=2)
    return np.unravel_index(np.argmin(dists), dists.shape)

def update_weights(X_batch, original_indices, W, distance_matrix, current_bmus, eta, sigma):
    delta_W = np.zeros_like(W)
    for x, original_idx in zip(X_batch, original_indices):
        dists = np.linalg.norm(W - x, axis=2)
        bmu = np.unravel_index(np.argmin(dists), dists.shape)
        bmu_weights = W[bmu]
        similarity_to_bmu = np.linalg.norm(W - bmu_weights, axis=2)
        similarity_threshold = 0.1
        similarity_mask = (similarity_to_bmu > similarity_threshold).astype(float)[:, :, np.newaxis]
        current_bmus[original_idx] = bmu


        distances = distance_matrix[bmu[0], bmu[1]]
        gaussian_part = np.exp(-distances**2 / (2 * sigma**2))[:, :, np.newaxis]  # Excitation (Gaussian part)
        inhibitory_part = np.exp(-distances**2 / (sigma**2))[:, :, np.newaxis]  # Inhibition part
        h = gaussian_part - inhibitory_part
        # r_squared = (distances / sigma)**2
        # h = (1 - r_squared) * np.exp(-r_squared / 2)
        # h = h[:, :, np.newaxis]
        h *= similarity_mask
        delta_W +=  h * (x-W)
    return W + eta * delta_W / X_batch.shape[0]


# =========================
# SOM Training
# =========================

def train_som_with_convergence_old(X, W, distance_matrix, n_epochs=100, sigma=1, eta=0.5, min_batch_size=1):
    m, k, dim = W.shape
    n_samples = X.shape[0]
    initial_eta, initial_sigma = eta, sigma
    batch_size = X.shape[0]//2
    prev_bmus = np.zeros((n_samples, 2), dtype=int)
    stats = {
        "bmu_movement": [],
        "bmu_switches": [],
        "quantization_error": [],
        "topographic_error": []
    }

    for epoch in range(n_epochs):
        bmu_influence = min(0, 1 / (1 + np.exp(-10 * (epoch/n_epochs - 0.5))))
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        current_bmus = np.zeros((n_samples, 2), dtype=int)
        
        for i in range(0, n_samples, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_samples))
            batch = X_shuffled[batch_slice]
            original_indices = shuffled_indices[batch_slice]
            W = update_weights(batch, original_indices, W, distance_matrix, current_bmus, eta, sigma)

        movement, switch_count, prev_bmus = compute_bmu_metrics(prev_bmus, current_bmus)
        stats["bmu_movement"].append(movement)
        stats["bmu_switches"].append(switch_count)
        stats["quantization_error"].append(quantization_error(W, X))
        stats["topographic_error"].append(topographic_error(W, X))
        
        switch_rate = switch_count / n_samples
        eta, sigma = update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma)
        if switch_rate > 0.5:
            eta *= 1.00
            sigma *= 0.98
        elif switch_rate > 0.2:
            eta *= 0.95
            sigma *= 0.95
        else:
            eta *= 0.90
            sigma *= 0.90

        if switch_rate < 0.3:
            batch_size = max(int(batch_size * 0.9), min_batch_size)
        else:
            batch_size = min(int(batch_size * 1.05), X.shape[0]//2)

        plot_rgb_som(W)
        
    return W, stats

def train_som_with_convergence(X, W, distance_matrix, n_epochs=100, sigma=1, eta=0.5, 
                               initial_batch_size=1000, min_batch_size=16, 
                               early_stop=True, stop_tolerance=1e-4, patience=3):
    
    m, k, dim = W.shape
    n_samples = X.shape[0]
    initial_eta, initial_sigma = eta, sigma
    batch_size = initial_batch_size
    prev_bmus = np.zeros((n_samples, 2), dtype=int)
    
    stats = {
        "bmu_movement": [],
        "bmu_switches": [],
        "quantization_error": [],
        "topographic_error": []
    }

    no_improve_count = 0  # for early stopping

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        current_bmus = np.zeros((n_samples, 2), dtype=int)

        # Adaptive learning rates will override this later
        eta, sigma = update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma)

        for i in range(0, n_samples, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_samples))
            batch = X_shuffled[batch_slice]
            original_indices = shuffled_indices[batch_slice]
            W = update_weights(batch, original_indices, W, distance_matrix, current_bmus, eta, sigma)

        # Compute metrics
        movement, switch_count, prev_bmus = compute_bmu_metrics(prev_bmus, current_bmus)
        stats["bmu_movement"].append(movement)
        stats["bmu_switches"].append(switch_count)
        qe = quantization_error(W, X)
        te = topographic_error(W, X)
        stats["quantization_error"].append(qe)
        stats["topographic_error"].append(te)

        # --- Adapt eta and sigma based on QE and TE ---
        if epoch > 1:
            delta_qe = abs(stats["quantization_error"][-1] - stats["quantization_error"][-2])
            delta_te = abs(stats["topographic_error"][-1] - stats["topographic_error"][-2])

            if delta_qe < stop_tolerance and delta_te < stop_tolerance:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if delta_qe > 0:
                eta *= 1.05  # QE increased? Try escaping by increasing learning rate
            else:
                eta *= 0.95

            if te > 0.2:
                sigma *= 1.05  # High TE? Encourage topology smoothing
            else:
                sigma *= 0.95

        # --- Batch size adaptation (as before) ---
        switch_rate = switch_count / n_samples
        if switch_rate < 0.3:
            batch_size = max(int(batch_size * 0.9), min_batch_size)
        else:
            batch_size = min(int(batch_size * 1.05), initial_batch_size)

        # Optional: early stop
        if early_stop and epoch >= 10 and no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch} due to QE/TE plateau.")
            break

        plot_rgb_som(W)

    return W, stats

# =========================
# Quality Metrics
# =========================

def compute_bmu_metrics(prev_bmus, current_bmus):
    grid_distances = np.abs(current_bmus - prev_bmus).sum(axis=1)
    mean_movement = np.mean(grid_distances)
    switch_flags = np.any(current_bmus != prev_bmus, axis=1)
    switch_count = np.sum(switch_flags)
    return mean_movement, switch_count, current_bmus.copy()

def quantization_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    return np.mean(np.min(cdist(X, flat_W), axis=1))

def topographic_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    codesize = W.shape[:2]
    errors = []

    for x in X:
        dists = np.linalg.norm(flat_W - x, axis=1)
        bmu1, bmu2 = np.argsort(dists)[:2]
        pos1, pos2 = np.unravel_index(bmu1, codesize), np.unravel_index(bmu2, codesize)
        errors.append(0 if np.linalg.norm(np.array(pos1) - np.array(pos2)) == 1 else 1)

    return np.mean(errors)

def compute_som_quality_metrics(X, W, current_bmus):
    qe = quantization_error(W, X)
    te = topographic_error(W, X)
    return qe, te

# =========================
# U-Matrix and Visualization
# =========================

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
    W = np.clip(W,a_min=0,a_max=1)
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
    red_samples = np.random.uniform(low=[0.8, 0.0, 0.0], high=[1, 0.2, 0.2], size=(n_red, 3))
    
    # Other colors (uniformly spread)
    other_samples = np.random.uniform(0, 1, size=(n_other, 3))

    X = np.vstack([red_samples, other_samples])
    np.random.shuffle(X)
    return X

# --- RGB Dataset ---
N = 2000
X_rgb_scaled = np.random.rand(N,3)
X_skewed = generate_skewed_rgb_dataset(n_samples=2000, red_ratio=0.8)

m, k = 9, 9
distance_matrix = precompute_distances(m, k)
som_shape = (m, k)  
# W = np.random.rand(m, k, X_skewed.shape[1])
# G = nx.grid_2d_graph(m, k)
# plot_rgb_som(W)
# start_time = time.time()
# # W_trained = train_som(X_rgb_scaled, W, G)
# # W_trained = train_som_optimized(X_rgb_scaled, W, distance_matrix)
# # W_trained, metrics = train_som_epochwise(X_rgb_scaled, W, distance_matrix)
# # W_trained, metrics = train_som_with_convergence(X_rgb_scaled, W, distance_matrix, n_epochs=100)
# W_trained, metrics = train_som_with_convergence(X_skewed, W, distance_matrix, n_epochs=100)
# end_time = time.time()
# total_time = end_time-start_time
# print(total_time)
# plot_rgb_som(W_trained)

# # bmu_locations = map_samples_to_nodes(X_rgb_scaled, W_trained)
# # plot_samples(X_rgb_scaled, bmu_locations)


W = np.random.rand(m, k, X_skewed.shape[1])
plot_rgb_som(W)
X = W.reshape(-1, 3)
# W_init, metrics = train_som_with_convergence_old(X_skewed, W, distance_matrix, sigma=1.5, eta=1,n_epochs=50)
# W_init = np.clip(W_init,a_min=0,a_max=1)


W_trained, metrics = train_som_with_convergence(X_skewed, W, distance_matrix, sigma=1, eta=0.5,n_epochs=100)
plot_rgb_som(W_trained)

# Plot Metrics maison
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BMU Movement Distance', color=color)
ax1.plot(metrics["bmu_movement"], color=color)
ax1.tick_params(axis='y', labelcolor=color) 
ax1.set_yscale("log")

color = 'tab:orange'
ax2 = ax1.twinx()  
ax2.set_ylabel('BMU Switch Count', color=color) 
ax2.plot(metrics["bmu_switches"], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale("log")

plt.title("SOM Convergence Metrics")
plt.grid(True)
fig.tight_layout() 
plt.show()


fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Quantization Error', color=color)
ax1.plot(metrics['quantization_error'], color=color)
ax1.tick_params(axis='y', labelcolor=color) 

color = 'tab:orange'
ax2 = ax1.twinx()  
ax2.set_ylabel('Topographic Error', color=color) 
ax2.plot(metrics['topographic_error'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("SOM Quality Metrics")
plt.grid(True)
fig.tight_layout() 
plt.show()



# som = somoclu.Somoclu(m, k, gridtype='rectangular')
# som.train(X_skewed)
# W = som.codebook.reshape((m, k, -1))
# plot_rgb_som(W)

# # UMAP
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# embedding_rgb = reducer.fit_transform(X_rgb_scaled)
# embedding_skewed = reducer.fit_transform(X_skewed)

# def plot_embedding(embedding, colors, title):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# # Plot original uniform RGB data
# plot_embedding(embedding_rgb, X_rgb_scaled, title='UMAP Embedding of Random RGB Data')

# # Plot skewed RGB data
# plot_embedding(embedding_skewed, X_skewed, title='UMAP Embedding of Skewed RGB Data')