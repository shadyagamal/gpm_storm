"""
Train a Self-Organizing Map (SOM) using patch statistics.

@author: shadya
"""

import os
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.som.experiments import get_experiment_info, save_som  # type: ignore
import numpy as np
from somperf.metrics import *
from somperf.utils.topology import rectangular_topology_dist
from minisom import MiniSom
from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.experiments import get_experiment_info, load_som
from gpm_storm.som.io import(
    create_dask_cluster,
    create_som_df_array,
    create_som_df_features_stats,
    create_som_sample_ds_array,
    sample_node_datasets,)
from gpm_storm.som.plot import(
    plot_images,)
import itertools

# Initial chosen vars
# ICC_30_max
# ICC_40_max
# LCC_30_max
# LCC_40_max
# CC_40_count
# CC_30_count
# P_max
# P_sum
# P_count
# MP_sum
# P_GT2_regions
# P_GT2_count
# P_GT10_regions
# P_GT10_count
# P_GT50_regions
# P_GT50_count
# P_GT120_regions
# P_GT120_count
# P_%_between_0_1
# P_%_between_5_10
# P_%_between_20_300


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
    df_na = df_selected.dropna(axis=1)
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_na),
        columns=df_na.columns,
        index=df_na.index,
    )

    print(f"Data preprocessing complete! {len(df) - len(df_scaled)} rows removed due to NaNs.\n")
    return df_cleaned, df_scaled, df_na


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
    row_values = range(10)  
    col_values = range(10)  
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
som_name = "Test_SOM"  
n_rows, n_columns = 10, 10

df = pd.read_parquet(filepath) 
vars = df.columns[:134]

df_cleaned, df_scaled, df_na = preprocess_data(df, vars)
som = train_som(df_na, som_name, som_dir, n_rows, n_columns)
bmus = som.bmus  

df_bmu = df_na.copy()
df_bmu["row"], df_bmu["col"] = bmus[:, 0], bmus[:, 1]
df_final = df.copy()
df_final.loc[df_bmu.index, "row"] = df_bmu["row"]
df_final.loc[df_bmu.index, "col"] = df_bmu["col"]

missing_combinations = check_missing_combos(df_final,n_rows, n_columns)

new_filepath = os.path.expanduser(f"~/gpm_storm/data/{som_name}_with_bmus.parquet")
df_final.to_parquet(new_filepath)

som = load_som(som_dir=som_dir, som_name=som_name)
