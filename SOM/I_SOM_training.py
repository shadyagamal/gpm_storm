r"""
Train a Self-Organizing Map (SOM) using patch statistics.

@author: shadya
"""
import os
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler
from gpm_storm.som.experiments import save_som
import itertools
import numpy as np
from collections import Counter
from scipy.stats import skew
from somperf.metrics import * #type: ignore
import umap
from sklearn.cluster import KMeans

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

def inverse_density_sample_df(df, sample_size):
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

def get_umap_kmeans_codebook(X, n_rows, n_columns, umap_dim=3, random_state=42):
    n_nodes = n_rows * n_columns
    reducer = umap.UMAP(n_components=umap_dim, random_state=random_state)
    X_umap = reducer.fit_transform(X)
    kmeans = KMeans(n_clusters=n_nodes, random_state=random_state)
    kmeans.fit(X_umap)
    labels = kmeans.labels_
    
    cluster_centroids = np.zeros((n_nodes, X.shape[1]), dtype=np.float32)
    for i in range(n_nodes):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            cluster_points = X[np.random.choice(len(X), 1)]
        cluster_centroids[i] = np.mean(cluster_points, axis=0)
    codebook = cluster_centroids.reshape(n_rows, n_columns, -1)
    return codebook

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

def preprocess_data_full(df):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.droppna(axis=1)
            
    df_selected = df_cleaned[df_cleaned.columns[-9]]
    df_rounded, _ = bin_and_round_df(df_selected)
    df_thresh = df_rounded[df_rounded["P_mean"]>=1]

    skewed_cols = df_thresh.apply(skew).pipe(lambda s: s[s > 0.75].index.tolist())
    df_log = df_thresh.copy()
    for col in skewed_cols:
        if (df_log[col] >= 0).all():
            df_log[col] = np.log1p(df_log[col])

    df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df_log), columns=df_log.columns, index=df_log.index)
    return df_scaled, df.iloc[df_thresh.index]


def train_som(df_scaled, initial_codebook, n_rows, n_columns, som_name, som_dir, epochs=100):
    # Initialize SOM
    som = somoclu.Somoclu(
        n_columns=n_columns, n_rows=n_rows,
        initialcodebook=initial_codebook,
        gridtype="rectangular", maptype="toroid"
    )

    # Train SOM
    som.train(
        data=df_scaled.to_numpy(),
        epochs=epochs,
        radius0=7.5, radiusN=1,
        radiuscooling='linear',
        scale0=1.1, scaleN=0.001,
        scalecooling='linear'
    )
    
    # Save the trained SOM
    save_som(som, som_dir=som_dir, som_name=som_name)
    print("SOM training complete and model saved!\n")
    return som

def check_missing_nodes(df_bmu, n_rows, n_columns):
    expected = set(itertools.product(range(n_rows), range(n_columns)))
    actual = set(zip(df_bmu["row"], df_bmu["col"], strict=False))
    missing = expected - actual
    print(f"Missing nodes: {missing}" if missing else "No missing nodes.")
    return missing

def precompute_distances(m, k):
    i1, j1, i2, j2 = np.indices((m, k, m, k))
    return np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)

def compute_metrics(som, df_scaled, dist_matrix):
    codebook = som.codebook.reshape((-1, df_scaled.shape[1]))
    qe = quantization_error(codebook, df_scaled.to_numpy())
    k = som._n_columns
    def dist_fun(u1, u2):
        i1, j1 = divmod(u1, k)
        i2, j2 = divmod(u2, k)
        return dist_matrix[i1, j1, i2, j2]
    te = topographic_error(dist_fun, codebook, df_scaled.to_numpy())
    tp = topographic_product(dist_fun, codebook)
    print(f"Quantization error: {qe:.4f}")
    print(f"Topographic error: {te:.4f}")
    print(f"Topographic product: {tp:.4f}")
    return qe, te, tp


# --- Load data ---
filepath0 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_0.parquet"
filepath1 = "/ltenas2/data/GPM_STORM_DB/merged/merged_data_total_1.parquet"
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")
os.makedirs(som_dir, exist_ok=True)

df0 = pd.read_parquet(filepath0) 
df1 = pd.read_parquet(filepath1) 
df = pd.concat([df0,df1], ignore_index=True)

# --- Variables to use ---
vars_to_use = [
    "ICC_30_max", "ICC_40_max", "LCC_30_max", "LCC_40_max", "CC_40_count", "CC_30_count", "P_mean",
    "P_max", "P_sum", "P_count", "MP_sum", "P_GT2_regions", "P_GT2_count", "P_GT10_regions","P_GT120_mean",
    "P_GT10_count", "P_GT50_regions", "P_GT50_count", "P_GT120_regions", "P_GT120_count",
    "P_%_between_0_1", "P_%_between_5_10", "P_%_between_20_300"
]

# --- Preprocessing ---
som_name = "SOM_Pmean_>_1_FULL_random"  
n_rows, n_columns = 10, 10
n_nodes = n_rows * n_columns
distance_matrix = precompute_distances(n_rows, n_columns)
df_scaled, df_original = preprocess_data_full(df, vars_to_use)
sample = df_scaled.sample(n=n_nodes, replace=False)

# --- Codebook Initializations ---
W = {
    "sampled": np.ascontiguousarray(sample.to_numpy().reshape(n_rows, n_columns, -1), dtype=np.float32),
    "random": np.random.normal(0, 1, size=(n_rows, n_columns, df_scaled.shape[1])).astype(np.float32),
    "umap": get_umap_kmeans_codebook(df_scaled, n_rows, n_columns)
}

# --- Training ---
som = train_som(df_scaled, W["umap"], n_rows, n_columns, som_name, som_dir, epochs=200)
# metrics = compute_metrics(som, df_scaled, distance_matrix)

# --- Checking Nodes ---
bmus = som.bmus
df_bmu = df_original.copy()
df_bmu["row"], df_bmu["col"] = bmus[:, 0], bmus[:, 1]
missing_nodes = check_missing_nodes(df_bmu, n_rows, n_columns)
new_filepath = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
df_bmu.to_parquet(new_filepath)





# node_positions = [(r, c) for r in range(n_rows) for c in range(n_columns)]
# X = df_full_scaled.to_numpy()
# distances = cdist(X, weights_2d, metric="euclidean")
# bmu_indices = np.argmin(distances, axis=1)
# bmu_coords = np.array([node_positions[i] for i in bmu_indices])
# df_full_bmu = df.copy()
# df_full_bmu["row"] = bmu_coords[:, 0]
# df_full_bmu["col"] = bmu_coords[:, 1]
# new_filepath = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
# df_bmu.to_parquet(new_filepath)
# # som = load_som(som_dir=som_dir, som_name=som_name)



