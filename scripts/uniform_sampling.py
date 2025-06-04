#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:12:36 2025

@author: gamal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import random


# --------------------------------------------
# RGB DATASET GENERATION AND VISUALIZATION
# --------------------------------------------
def generate_skewed_rgb_dataset(n_samples=1000, red_ratio=0.9):
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


def plot_rgb_grid(X, grid_size=(10, 10), title="RGB Sample Grid"):
    assert len(X) == grid_size[0] * grid_size[1], "Grid size must match number of samples"
    
    # Reshape for image display
    image = X.reshape(grid_size[0], grid_size[1], 3)
    image = np.clip(image, 0, 1)  # Ensure RGB values are in range

    plt.figure(figsize=(6, 6))
    plt.imshow(image, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def plot_histogram(df, dim1=5, dim2=5):
    fig, axes = plt.subplots(dim1, dim2, figsize=(20, 10))  
    axes = axes.flatten()
    for i, stat in enumerate(df.columns):  # <== Fix here
        sns.histplot(df[stat], bins=100, ax=axes[i], stat="probability")
        axes[i].set_title(stat)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Relative Frequency")
        axes[i].set_yscale("log")
    plt.tight_layout()
    plt.show()
    
# --------------------------------------------
# QUANTIZATION METHODS
# --------------------------------------------

def bin_and_round(X, bins=10):
    X = np.array(X)
    rounded = np.zeros_like(X)
    edges = []

    for i in range(X.shape[1]):
        col = X[:, i]
        col_min, col_max = col.min(), col.max()
        bin_edges = np.linspace(col_min, col_max, bins + 1)
        mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(col, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, bins - 1)
        rounded[:, i] = mids[bin_idx]
        edges.append(bin_edges)

    return rounded, edges

def inverse_density_sampling(X, bins=10, sample_size=100):
    hist, edges = np.histogramdd(X, bins=[bins]*X.shape[1], range=[[0, 1]]*X.shape[1])
    bin_idx = np.stack([
        np.digitize(X[:, d], edges[d]) - 1 for d in range(X.shape[1])
    ], axis=1)
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    densities = np.array([hist[tuple(idx)] for idx in bin_idx])
    probabilities = 1 / (densities + 1e-8)
    probabilities /= probabilities.sum()

    sampled_indices = np.random.choice(len(X), size=sample_size, replace=False, p=probabilities)
    return X[sampled_indices], edges

# --------------------------------------------
# RGB TEST + DEMO
# --------------------------------------------

X_rgb = generate_skewed_rgb_dataset(n_samples=200000, red_ratio=0.8)

# Regular random and inverse-weighted sampling
X_random = X_rgb[np.random.choice(len(X_rgb), size=100, replace=False)]
X_weighted, _ = inverse_density_sampling(X_rgb, bins=5, sample_size=100)

plot_rgb_grid(X_random, grid_size=(10, 10), title="Random RGB Sampling")
plot_rgb_grid(X_weighted, grid_size=(10, 10), title="Inverse-Weighted RGB Sampling")

# Bin + round
X_rounded, _ = bin_and_round(X_rgb, bins=10)
X_unique = np.unique(X_rounded, axis=0)
sampled_indices = np.random.choice(len(X_unique), size=100, replace=False)
plot_rgb_grid(X_unique[sampled_indices], grid_size=(10, 10), title="Binned + Rounded RGB Samples")
#------------------------------------------------------------------------------
filepath = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
df = pd.read_parquet(filepath) 

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
        
vars = ["ICC_30_max","ICC_40_max","LCC_30_max","LCC_40_max",
        "CC_40_count","CC_30_count",
        "P_max","P_sum","P_count","MP_sum",
        "P_GT2_regions","P_GT2_count","P_GT10_regions","P_GT10_count",
        "P_GT50_regions","P_GT50_count","P_GT120_regions","P_GT120_count",
        "P_%_between_0_1","P_%_between_5_10","P_%_between_20_300"
]
df_selected = df_cleaned[vars]

# Cutoff "outliers"
upper_thresholds = df_selected.quantile(0.99)
upper_thresholds["P_max"] = 300
upper_thresholds["P_%_between_0_1"] = 100
df_trimmed = df_selected[df_selected <= upper_thresholds].dropna()
plot_histogram(df_trimmed, dim1=5, dim2=5)


# === Bin and round (Doane binning per feature) ===
X_df = df_trimmed.copy()
binned_df = pd.DataFrame(index=X_df.index)
rounded_df = pd.DataFrame(index=X_df.index)
bin_edges = {}

for col in X_df.columns:
    hist, edges = np.histogram(X_df[col],bins="doane")
    binned_df[col] = np.digitize(X_df[col], bins=edges, right=False) - 1
    binned_df[col] = np.clip(binned_df[col], 0, len(edges) - 2)
    bin_edges[col] = edges
    bin_floors = edges[:-1]
    rounded_values = bin_floors[binned_df[col]]
    rounded_df[col] = rounded_values


counts = rounded_df.value_counts()
rounded_unique_df = rounded_df.drop_duplicates()
plot_histogram(rounded_unique_df, dim1=5, dim2=5)

# === Get unique binned samples and inverse-probability weights ===
from collections import Counter
sample_size = 10000

X_df = df_trimmed.copy()
binned_df = pd.DataFrame(index=X_df.index)
bin_edges = {}

for col in X_df.columns:
    data = X_df[col]
    hist, edges = np.histogram(data, bins="doane")
    bin_idx = np.digitize(data, bins=edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    binned_df[col] = bin_idx


bin_idx_tuples = [tuple(row) for row in binned_df.values]
bin_counts = Counter(bin_idx_tuples)


densities = np.array([bin_counts[tuple(row)] for row in binned_df.values])
probabilities = 1.0 / (densities + 1e-8)
probabilities /= probabilities.sum()

n_samples = min(sample_size, X_df.shape[0])
sampled_indices = np.random.choice(X_df.index, size=n_samples * 3, replace=False, p=probabilities)
sampled_df = X_df.loc[sampled_indices].drop_duplicates().head(n_samples)
