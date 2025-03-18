#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and visualize Kmeans clustering from patch statistics.

@author: shadya
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FILEPATH = os.path.expanduser("~/gpm_storm/data/patch_statistics.parquet") # f"feature_{granule_id}.parquet"

def preprocess_data(df):
    """Clean and standardize the dataset for K-Means."""
    print("Preprocessing data...")

    # Drop rows with missing values (change to axis=0)
    df_cleaned = df.dropna(axis=1)

    # Keep only numerical columns
    df_cleaned = df_cleaned.select_dtypes(include=[np.number])

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned)

    print("Data preprocessing complete!\n")
    return df_scaled

def perform_kmeans(data, n_clusters=3):
    """
    Perform K-Means clustering and return model, labels, and cluster centers.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)

    print("K-Means clustering complete!\n")
    return kmeans, kmeans.labels_, kmeans.cluster_centers_

def plot_clusters(data, labels, cluster_centers, n_clusters):
    """
    Plot K-Means clustering results.
    """
    print("Plotting clustering results...")

    plt.figure(figsize=(8, 6))

    # Scatter plot of data points with different colors for each cluster
    for i in range(n_clusters):
        plt.scatter(
            data[labels == i, 0], data[labels == i, 1], label=f"Cluster {i + 1}", alpha=0.5
        )

    # Plot cluster centers
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], c="black", marker="x", s=100, label="Centroids"
    )

    plt.title("K-Means Clustering")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    df = pd.read_parquet(FILEPATH)
    df_scaled = preprocess_data(df)
    kmeans, labels, cluster_centers = perform_kmeans(df_scaled)

    # Visualization
    plot_clusters(df_scaled, labels, cluster_centers, n_clusters=3)

if __name__ == "__main__":
    main()
