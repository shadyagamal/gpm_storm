r"""
Train a Self-Organizing Map (SOM) using patch statistics.

@author: shadya
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


def preprocess_data(df, vars):
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
    return df_scaled, df_original


def train_som(df_scaled, som_name, som_dir, n_rows=10, n_columns=10):
    """
    Train a Self-Organizing Map (SOM) using the selected features.
    """
    data = df_scaled.to_numpy()

    
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
    print("SOM training complete and model saved!\n")
    return som

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


filepath0 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
filepath1 = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet") 
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
som_name = "SOM_Pmean_>_1_uniform"  
n_rows, n_columns = 10, 10
n_nodes = n_rows * n_columns

df0 = pd.read_parquet(filepath0) 
df1 = pd.read_parquet(filepath1) 
df = pd.concat([df0,df1], ignore_index=True)

vars = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count", "P_mean",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]

# df_cleaned, df_rounded, df_scaled, df_full_scaled = preprocess_data(df, vars)
df_scaled, df_original = preprocess_data(df, vars)
som = train_som(df_scaled, som_name, som_dir, n_rows, n_columns)
bmus = som.bmus
df_bmu = df_original.copy()
df_bmu["row"], df_bmu["col"] = bmus[:, 0], bmus[:, 1]
missing_combinations = check_missing_combos(df_bmu,n_rows, n_columns)

# weights = som.codebook
# weights_2d = weights.reshape((n_nodes, -1))
# node_positions = [(r, c) for r in range(n_rows) for c in range(n_columns)]
# X = df_full_scaled.to_numpy()
# distances = cdist(X, weights_2d, metric="euclidean")
# bmu_indices = np.argmin(distances, axis=1)
# bmu_coords = np.array([node_positions[i] for i in bmu_indices])
# df_full_bmu = df.copy()
# df_full_bmu["row"] = bmu_coords[:, 0]
# df_full_bmu["col"] = bmu_coords[:, 1]


new_filepath = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
df_bmu.to_parquet(new_filepath)
# som = load_som(som_dir=som_dir, som_name=som_name)
