"""
Train a Self-Organizing Map (SOM) using patch statistics.

@author: shadya
"""

import os
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.som.experiments import get_experiment_info, save_som  # type: ignore
from gpm_storm.som.som_metrics import quantization_error, topographic_product
import numpy as np
from somperf.metrics import *
from somperf.utils.topology import rectangular_topology_dist
from minisom import MiniSom


filepath = ("/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet") 
df = pd.read_parquet(filepath) 
som_dir = os.path.expanduser("~/gpm_storm/scripts")  
som_name = "zonal_SOM"  
vars = df.columns[0:-9]


def preprocess_data(df, features):
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
            df[col] = df_cleaned[col].fillna(0)
    df_cleaned = df.dropna(axis=1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned[features]),
        columns=features,
        index=df_cleaned.index,
    )
    print(f"Data preprocessing complete! {len(df) - len(df_cleaned)} rows removed due to NaNs.\n")
    return df_scaled


def train_som(df_scaled, som_name):
    """
    Train a Self-Organizing Map (SOM) using the selected features.
    """
    # n_rows, n_columns = 10, 10
    data = df_scaled.to_numpy()
    X = data
    map_size = (8, 10)
    # Initialize SOM
    # som = somoclu.Somoclu(
    #     n_columns=n_columns,
    #     n_rows=n_rows,
    #     gridtype="rectangular",
    #     maptype="planar",
    # )

    # # Train SOM
    # som.train(
    #     data=data,
    #     epochs=100,
    #     radius0=0,
    #     radiusN=1,
    #     scale0=0.5,
    #     scaleN=0.001,
    # )
    som = MiniSom(map_size[0], map_size[1], X.shape[-1], sigma=1.0, learning_rate=1.0, random_seed=42)
    som.random_weights_init(X)
    som.train_random(X, 10000)
    weights = som.get_weights().reshape(map_size[0]*map_size[1], -1)
    print('Topographic product = ', topographic_product(rectangular_topology_dist(map_size), weights))
    
    # Save the trained SOM
    # save_som(som, som_dir=som_dir, som_name=som_name)
    # print("SOM training complete and model saved!\n")

def train_som(df_scaled, som_name=None, search_range=(5, 15), max_epochs=10000):
    """
    Train a Self-Organizing Map (SOM) using the selected features,
    and find the optimal map size using the topographic product.
    """
    X = df_scaled.to_numpy()
    input_len = X.shape[-1]

    best_P = np.inf
    best_map_size = None
    best_som = None

    print("Searching for best map size using topographic product...")
    for rows in range(search_range[0], search_range[1]+1):
        for cols in range(search_range[0], search_range[1]+1):
            som = MiniSom(rows, cols, input_len, sigma=1.0, learning_rate=1.0, random_seed=42)
            som.random_weights_init(X)
            som.train_random(X, max_epochs)

            weights = som.get_weights().reshape(rows * cols, -1)
            P = topographic_product(rectangular_topology_dist((rows, cols)), weights)

            print(f"Map size {rows}x{cols} => Topographic product: {P:.4f}")
            if abs(P) < abs(best_P):
                best_P = P
                best_map_size = (rows, cols)
                best_som = som

    print(f"\nBest map size: {best_map_size} with P = {best_P:.4f}")

    # Optionally save the best SOM
    # if som_name:
    #     save_som(best_som, som_dir=som_dir, som_name=som_name)

    return best_som, best_map_size, best_P

def main():
    df = pd.read_parquet(filepath)

    # Get experiment features
    # experiment_info = get_experiment_info(SOM_NAME)
    # features = experiment_info["features"]

    df_scaled = preprocess_data(df, vars)
    train_som(df_scaled, som_name)
    train_som(df_scaled, som_name=None, search_range=(100, 101), max_epochs=100)

if __name__ == "__main__":
    main()
