#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:31:30 2025

@author: gamal
"""

import os
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.som.experiments import save_som, load_som
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from scipy.stats import skew
import matplotlib.pyplot as plt

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

def inverse_density_sample_df(df, sample_size=500000):
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
    return df.loc[sampled_idx]

def check_missing_combos(df,n_rows=10, n_columns=10):
    row_values = range(n_rows)  
    col_values = range(n_columns)  
    expected_combinations = set(itertools.product(row_values, col_values))
    actual_combinations = set(zip(df["row"], df["col"], strict=False))
    missing_combinations = expected_combinations - actual_combinations
    if missing_combinations:
        print(f"Missing nodes: {missing_combinations}")
    else:
        print("No missing (row, col) combinations.\n")
    return missing_combinations


filepath = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
som_dir = os.path.expanduser("~/gpm_storm/SOM/trained_soms/")  
som_name = "testing_SOM"  
n_rows, n_columns = 10, 10
n_nodes = n_rows * n_columns

df = pd.read_parquet(filepath) 
vars = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]
# vars = df.columns[:134]

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

df_cleaned = df_cleaned[df_cleaned["P_mean"]>1]

df_selected = df_cleaned[vars]
# df_selected = df_selected.dropna(axis=0)

df_selected[df_selected.columns].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

df_rounded = df_selected.copy()
# df_rounded = inverse_density_sample_df(df_rounded, sample_size=350000)
skew_vals = df_rounded.apply(skew, nan_policy='omit')
skewed_cols = skew_vals[skew_vals > 0.75].index.tolist()
for col in skewed_cols:
    if (df_rounded[col] >= 0).all():  # Only log-transform if values are non-negative
        df_rounded[col] = np.log1p(df_rounded[col])



scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_rounded),
    columns=df_rounded.columns,
    index=df_rounded.index,
    )

data = df_scaled.to_numpy()


df_scaled[df_scaled.columns].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()


# Initialize SOM
som = somoclu.Somoclu(
    n_columns=n_columns,
    n_rows=n_rows,
    gridtype="rectangular",
    maptype="planar",
)

# Train SOM
som.train(
    data=data,
    epochs=100,
    radius0=0,
    radiusN=1,
    radiuscooling='linear',
    scale0=0.5,
    scaleN=0.001,
    scalecooling='linear'
)

# Save the trained SOM
save_som(som, som_dir=som_dir, som_name=som_name)
bmus = som.bmus
df_bmu = df_scaled.copy()
df_bmu["row"], df_bmu["col"] = bmus[:, 0], bmus[:, 1]
missing_combinations = check_missing_combos(df_bmu,n_rows, n_columns)


counts = Counter(map(tuple, som.bmus))
for row in range(n_rows):
    for col in range(n_columns):
        print(f"Node ({row}, {col}): {counts.get((row, col), 0)} BMUs")

