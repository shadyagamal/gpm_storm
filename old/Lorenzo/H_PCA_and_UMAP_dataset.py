#!/usr/bin/env python3
"""

@author: shadya
"""

import os

import pandas as pd
from C_CorrelationMatrix import plot_correlation_matrix
from D_PCA import perform_pca, preprocess_data
from sklearn.cluster import KMeans

FILEPATH = os.path.expanduser("~/gpm_storm/data/largest_patch_statistics.parquet")  # f"feature_{granule_id}.parquet"


def perform_kmeans(df_pca, n_clusters=5):
    """
    Perform K-Means clustering on PCA results.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_pca["Cluster"] = kmeans.fit_predict(df_pca)

    print("K-Means clustering complete!\n")
    return kmeans, df_pca


def analyze_clusters(df_scaled, df_pca):
    """
    Analyze clusters and compute mean values per cluster.
    """
    print("Analyzing clusters...")

    # Merge the original scaled data with cluster assignments
    df_with_clusters = pd.concat([df_scaled, df_pca["Cluster"]], axis=1)

    # Calculate mean values for each cluster
    cluster_means = df_with_clusters.groupby("Cluster").mean()

    print("Cluster means computed!\n")
    print(cluster_means)


def display_pca_loadings(pca, df_scaled):
    """
    Display PCA loadings (feature contributions to principal components).
    """
    print("Computing PCA loadings...")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)],
        index=df_scaled.columns,
    )

    print("PCA loadings computed!\n")
    print(loadings)


def main():
    df = pd.read_parquet(FILEPATH)
    df_scaled = preprocess_data(df)

    # Perform PCA
    pca, df_pca = perform_pca(df_scaled, variance_threshold=0.95)

    # Visualize PCA correlation
    plot_correlation_matrix(df_pca)

    # Perform K-Means clustering
    kmeans, df_pca = perform_kmeans(df_pca, n_clusters=5)

    # Analyze clusters
    analyze_clusters(df_scaled, df_pca)

    # Display PCA loadings
    display_pca_loadings(pca, df_scaled)


if __name__ == "__main__":
    main()
