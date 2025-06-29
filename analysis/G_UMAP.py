#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 09:38:54 2025

@author: gamal
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import skew
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib.colors import ListedColormap

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
    return rounded_df.drop_duplicates(), bin_edges

def preprocess_data(df, vars):
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

    skewed_cols = df_thresh.apply(skew).pipe(lambda s: s[s > 0.75].index.tolist())
    df_log = df_thresh.copy()
    for col in skewed_cols:
        if (df_log[col] >= 0).all():
            df_log[col] = np.log1p(df_log[col])

    df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_log), columns=df_log.columns, index=df_log.index)
    return df_scaled, df.iloc[df_thresh.index]

# --- Load data ---
filepath0 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
filepath1 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet"
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")
som_name = "SOM_Pmean_>_1_with_random_init"  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
os.makedirs(som_dir, exist_ok=True)

df0 = pd.read_parquet(filepath0) 
df1 = pd.read_parquet(filepath1) 
df = pd.concat([df0,df1], ignore_index=True)
df_bmu = pd.read_parquet(bmu_dir)
df_bmu["som_cluster"] = df_bmu["row"] * 10 + df_bmu["col"]

# --- Variables to use ---
vars_to_use = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count", "P_mean",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions","P_GT120_mean",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]
df_scaled, df_original = preprocess_data(df, vars_to_use)

# --- UMAP ---
reducer = umap.UMAP()
embedding = reducer.fit_transform(df_scaled)


palette = sns.color_palette("gist_rainbow", n_colors=100)
cmap_100 = ListedColormap(palette)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=df_bmu["som_cluster"],
    cmap=cmap_100,
    alpha=0.7,
    s=10
)
plt.colorbar(scatter, label="SOM Cluster ID")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection Colored by SOM Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()



from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=10).fit(embedding)
df_bmu["umap_cluster"] = db.labels_ 

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=df_bmu["umap_cluster"],
    cmap=cmap_100,
    alpha=0.7,
    s=10
)
plt.colorbar(scatter, label="SOM Cluster ID")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection Colored by SOM Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()


