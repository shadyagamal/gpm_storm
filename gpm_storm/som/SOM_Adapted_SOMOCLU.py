#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:59:50 2025

@author: gamal
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
from gpm_storm.som.experiments import save_som, load_som
import itertools
from collections import Counter
from scipy.stats import skew
from tqdm import tqdm
from itertools import product
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

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

def bin_and_round_df(df):
    binned_df = pd.DataFrame(index=df.index)
    rounded_df = pd.DataFrame(index=df.index)
    bin_edges = {}
    for col in df.columns:
        hist, edges = np.histogram(df[col], bins="auto")
        bin_idx = np.digitize(df[col], bins=edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
        bin_edges[col] = edges
        binned_df[col] = bin_idx
        rounded_df[col] = edges[:-1][bin_idx]
    rounded_df = rounded_df.drop_duplicates()
    return rounded_df, bin_edges

def update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma):
    # Exponential
    # decay = np.exp(-*epoch / (n_epochs / 2))
    # eta = initial_eta * decay
    # sigma = initial_sigma * np.exp(-epoch / (n_epochs * 0.75))
    # Linear
    eta = initial_eta * (1 - 3*epoch / n_epochs)
    sigma = initial_sigma * (1 - 3*epoch / n_epochs)
    # eta = initial_eta
    # sigma = initial_sigma
    return eta, sigma

# def update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma):
#     min_eta = 0.05
#     min_sigma = 0.8 
#     eta = max(min_eta, initial_eta * (1 - epoch / n_epochs))
#     sigma = max(min_sigma, initial_sigma * (1 - epoch / n_epochs))
#     return eta, sigma

def inverse_density_sample_df(df, sample_size=100):
    binned_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        hist, edges = np.histogram(df[col])
        bin_idx = np.digitize(df[col], bins=edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
        binned_df[col] = bin_idx
    bin_tuples = [tuple(row) for row in binned_df.values]
    counts = Counter(bin_tuples)
    densities = np.array([counts[tuple(row)] for row in binned_df.values])
    probabilities = 1.0 / (densities + 1e-8)
    probabilities /= probabilities.sum()
    sampled_idx = np.random.choice(df.index, size=sample_size, replace=False, p=probabilities)
    df_sampled = df.loc[sampled_idx]
    
    scaler = MinMaxScaler()
    df_sampled = pd.DataFrame(
        scaler.fit_transform(df_sampled),
        columns=df_sampled.columns,
        index=df_sampled.index,
    )
    return df_sampled


def mean_inter_neuron_distance(codebook):
    """Compute the average distance between each neuron and all others."""
    codebook = codebook.reshape(-1, codebook.shape[-1])
    dists = cdist(codebook, codebook, metric='euclidean')
    return np.mean(dists)

def silhouette_score_som(X, codebook):
    """Compute silhouette score using BMU assignments as labels."""
    codebook = codebook.reshape(-1, codebook.shape[-1])
    labels = pairwise_distances_argmin(X, codebook)
    if len(np.unique(labels)) < 2:
        return -1  # silhouette_score requires at least 2 clusters
    return silhouette_score(X, labels)

def topographic_product(codebook, n_rows, n_columns):
    """Estimate topographic product (simplified)."""
    # compute distances to first and second BMUs
    codebook = codebook.reshape(-1, codebook.shape[-1])
    distances = pairwise_distances(codebook)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :2]
    prod = distances[np.arange(len(distances)), nearest[:, 0]] * distances[np.arange(len(distances)), nearest[:, 1]]
    return np.mean(prod)

def preprocess_data(df, vars, n_rows, n_columns):
    """Preprocess dataset by filtering NaNs and normalizing."""
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
            df_cleaned[col] = df_cleaned[col].fillna(0)
            
    df_selected = df_cleaned[vars]
    df_rounded, _ = bin_and_round_df(df_selected)
    df_thresh = df_rounded[df_rounded["P_mean"]>=1]
    
    df_uniform = inverse_density_sample_df(df_thresh, sample_size = n_rows*n_columns)
    
    df_log = df_thresh.copy()
    skew_vals = df_log.apply(skew, nan_policy='omit')
    skewed_cols = skew_vals[skew_vals > 0.75].index.tolist()
    for col in skewed_cols:
        if (df_log[col] >= 0).all():
            df_log[col] = np.log1p(df_log[col])
            
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_log),
        columns=df_log.columns,
        index=df_log.index,
    )
    df_original = df.iloc[df_thresh.index]
    return df_scaled, df_original, df_uniform

def train_som_with_convergence(X, initialcodebook, n_epochs=50, sigma=1, eta=0.5,  
                               early_stop=True, stop_tolerance=1e-4, patience=3):

    n_rows, n_columns, dim = initialcodebook.shape
    n_samples = X.shape[0]
    initial_eta, initial_sigma = eta, sigma
    prev_bmus = np.zeros((n_samples, 2), dtype=int)
    W = initialcodebook
    stats = {
        "bmu_movement": [],
        "bmu_switches": [],
        "quantization_error": [],
        "topographic_error": []
    }

    for epoch in tqdm(range(n_epochs)):
        eta, sigma = update_learning_rates(epoch, n_epochs, initial_eta, initial_sigma)
        som = somoclu.Somoclu(
            n_columns=n_columns,
            n_rows=n_rows,
            initialcodebook=W,
            gridtype="rectangular",
            maptype="planar",
        )
        
        # Train SOM
        som.train(data=X, epochs=3, radius0=sigma, scale0=eta,)
        W = som.codebook
        current_bmus = som.bmus
        # Compute metrics
        movement, switch_count, prev_bmus = compute_bmu_metrics(prev_bmus, current_bmus)
        stats["bmu_movement"].append(movement)
        stats["bmu_switches"].append(switch_count)
        qe = quantization_error(W, X)
        te = topographic_error(W, X)
        stats["quantization_error"].append(qe)
        stats["topographic_error"].append(te)
        
    return W, stats, som

def evaluate_som(X, n_rows, n_columns, sigma, eta, n_epochs, codebook):
    
    W, metrics, som = train_som_with_convergence(X, codebook, n_epochs=n_epochs, sigma=sigma, eta=eta)
    
    plt.plot(metrics['quantization_error'], label='QE')
    plt.plot(metrics['topographic_error'], label='TE')
    plt.legend()
    plt.title(f"SOM Quality Metrics - sigma:{sigma}, eta:{eta}")
    plt.grid(True)
    plt.show()
    
    final_qe = metrics["quantization_error"][-1]
    final_te = metrics["topographic_error"][-1]
    qe = metrics["quantization_error"]
    te = metrics["topographic_error"]
    score = [q + t for q, t in zip(qe, te)]
    
    mind = mean_inter_neuron_distance(W)
    sil_score = silhouette_score_som(X, W)
    tp = topographic_product(W, n_rows, n_columns)
    print(f"QE: {final_qe:.4f}, TE: {final_te:.4f}, MIND: {mind:.4f}, Silhouette: {sil_score:.4f}, TP: {tp:.4f}")
    
    return qe , te, mind, sil_score, tp, score, som

best_score = float("inf")
best_params = None

filepath0 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
filepath1 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet") 
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/") 
os.makedirs(som_dir, exist_ok=True)  

df0 = pd.read_parquet(filepath0) 
df1 = pd.read_parquet(filepath1) 
df = pd.concat([df0,df1], ignore_index=True)

# --- Variables to use ---
vars = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count", "P_mean",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count","P_GT120_mean",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]

# --- Preprocessing ---
som_name = "Optimized_SOM"  
n_rows, n_columns = 10, 10
n_nodes = n_rows * n_columns
som_shape = (n_rows, n_columns) 
df_scaled, df_original, df_uniform = preprocess_data(df, vars, n_rows, n_columns)
X = df_scaled.to_numpy().astype(np.float32)


initialcodebook_uni = np.ascontiguousarray(
    df_uniform.to_numpy().reshape(n_rows, n_columns, X.shape[1]), dtype=np.float32)

np.random.seed(42)
initialcodebook_random = np.random.normal(0, 1, size=(n_rows, n_columns, X.shape[1]))
initialcodebook_random = initialcodebook_random.astype(np.float32)

sampled_idx = np.random.choice(df_scaled.index, size=100, replace=False)
df_sampled = df_scaled.loc[sampled_idx]
initialcodebook_sampled = np.ascontiguousarray(
    df_sampled.to_numpy().reshape(n_rows, n_columns, X.shape[1]), dtype=np.float32)

param_grid = {
    "n_rows": [10],
    "n_columns": [10],
    "sigma": [7.5],
    "eta": [1.1],
    "n_epochs": [50],
}

for n_rows, n_columns, sigma, eta, n_epochs in product(
        param_grid["n_rows"],
        param_grid["n_columns"],
        param_grid["sigma"],
        param_grid["eta"],
        param_grid["n_epochs"]):
    
    qe , te, mind, sil_score, tp, score, som  = evaluate_som(X, n_rows, n_columns, sigma, eta, n_epochs, initialcodebook_sampled)
    min_score = min(score)
    min_index = score.index(min_score)
    print(min_score)
    if min_score < best_score:
        best_score = min_score
        best_params = (n_rows, n_columns, sigma, eta, n_epochs, min_index)
        print(f"New best score: {best_score:.4f} with params: {best_params}")

print("Best score:", best_score)
print("Best parameters:", best_params)


bmus = som.bmus
df_bmu = df_original.copy()
df_bmu["row"], df_bmu["col"] = bmus[:, 0], bmus[:, 1]
missing_combinations = check_missing_combos(df_bmu,n_rows, n_columns)









# df_mean1 = df[df["P_mean"]>1]
# from scipy.stats import chisquare
# numeric_cols = df_mean1.select_dtypes(include=np.number).columns
# chi = []
# for col in numeric_cols:
#     values = df_mean1[col].dropna()
#     hist, _ = np.histogram(values, bins=10)
#     expected = np.full(len(hist), fill_value=hist.sum() / len(hist))
#     chi2, p = chisquare(f_obs=hist, f_exp=expected)
#     chi.append(chi2)
#     print(f"{col}Chi-square: {chi2:.2f}, p-value: {p:.4f}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    