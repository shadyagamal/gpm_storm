#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:02:56 2025

@author: gamal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:12:36 2025
@author: gamal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random

# --------------------------------------------
# RGB DATA GENERATION AND VISUALIZATION
# --------------------------------------------

def generate_skewed_rgb_dataset(n_samples=1000, red_ratio=0.9):
    n_red = int(n_samples * red_ratio)
    n_other = n_samples - n_red
    red_samples = np.random.uniform(low=[0.8, 0.0, 0.0], high=[1.0, 0.2, 0.2], size=(n_red, 3))
    other_samples = np.random.uniform(0, 1, size=(n_other, 3))
    X = np.vstack([red_samples, other_samples])
    np.random.shuffle(X)
    return X

def plot_rgb_grid(X, grid_size=(10, 10), title="RGB Sample Grid"):
    assert len(X) == grid_size[0] * grid_size[1]
    image = X.reshape(grid_size[0], grid_size[1], 3)
    image = np.clip(image, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_histogram(df, dim1=5, dim2=5):
    fig, axes = plt.subplots(dim1, dim2, figsize=(20, 10))  
    axes = axes.flatten()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], stat="probability")
        axes[i].set_title(col)
        axes[i].set_yscale("log")
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# SAMPLING + BINNING METHODS
# --------------------------------------------

def inverse_density_sample_df(df, sample_size=10000):
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
    sampled_idx = np.random.choice(df.index, size=sample_size * 3, replace=False, p=probabilities)
    return df.loc[sampled_idx].drop_duplicates().head(sample_size)

def bin_and_round_df(df):
    binned_df = pd.DataFrame(index=df.index)
    rounded_df = pd.DataFrame(index=df.index)
    bin_edges = {}
    for col in df.columns:
        hist, edges = np.histogram(df[col], bins="doane")
        bin_idx = np.digitize(df[col], bins=edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
        bin_edges[col] = edges
        binned_df[col] = bin_idx
        rounded_df[col] = edges[:-1][bin_idx]
    rounded_df = rounded_df.drop_duplicates()
    return rounded_df, bin_edges

# --------------------------------------------
# TEST ON RGB DATASET
# --------------------------------------------

X_rgb = generate_skewed_rgb_dataset(n_samples=200_000, red_ratio=0.8)
X_random = X_rgb[np.random.choice(len(X_rgb), size=100, replace=False)]
plot_rgb_grid(X_random, grid_size=(10, 10), title="Random RGB Sampling")

X_weighted = inverse_density_sample_df(pd.DataFrame(X_rgb, columns=["R", "G", "B"]), sample_size=100)
plot_rgb_grid(X_weighted.values, grid_size=(10, 10), title="Inverse-Weighted RGB Sampling")

rounded_rgb_df, _ = bin_and_round_df(pd.DataFrame(X_rgb, columns=["R", "G", "B"]))
rounded_rgb_sample = rounded_rgb_df.sample(n=100, replace=False)
plot_rgb_grid(rounded_rgb_sample.values, grid_size=(10, 10), title="Rounded RGB Sampling")

# --------------------------------------------
# TEST ON GPM DATASET
# --------------------------------------------

filepath = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
df = pd.read_parquet(filepath)

fill_zero_cols = [
    "P_GT1_mean", "P_GT1_sum", "MA_LP_GT_0", "MiA_LP_GT_0", "MA_LP_GT_1", "MiA_LP_GT_1",
    "P_GT2_mean", "P_GT2_sum", "MA_LP_GT_2", "MiA_LP_GT_2", "P_GT5_mean", "P_GT5_sum",
    "MA_LP_GT_5", "MiA_LP_GT_5", "P_GT10_mean", "P_GT10_sum", "MA_LP_GT_10", "MiA_LP_GT_10",
    "P_GT20_mean", "P_GT20_sum", "MA_LP_GT_20", "MiA_LP_GT_20", "P_GT50_mean", "P_GT50_sum",
    "P_GT80_mean", "P_GT80_sum", "P_GT120_mean", "P_GT120_sum", "LCC_30_mean", "LCC_30_std",
    "ICC_30_mean", "ICC_30_std", "LCC_40_mean", "LCC_40_std", "ICC_40_mean", "ICC_40_std",
    "LCC_30_max", "ICC_30_max", "LCC_40_max", "ICC_40_max"
]

df_cleaned = df.copy()
for col in fill_zero_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna(0)

vars = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]

df_selected = df_cleaned[vars]
upper_thresholds = df_selected.quantile(0.99)
upper_thresholds["P_max"] = 300
upper_thresholds["P_%_between_0_1"] = 100
df_trimmed = df_selected[df_selected <= upper_thresholds].dropna()

# Plot original trimmed histograms
plot_histogram(df_trimmed, dim1=5, dim2=5)

# Round + plot
rounded_gpm_df, _ = bin_and_round_df(df_trimmed)
plot_histogram(rounded_gpm_df, dim1=5, dim2=5)

# Inverse-density sample + plot
sampled_gpm_df = inverse_density_sample_df(df_trimmed, sample_size=10000)
plot_histogram(sampled_gpm_df, dim1=5, dim2=5)
