#!/usr/bin/env python3
"""
Compute and visualize Kmeans clustering from patch statistics.

@author: shadya
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FILEPATH = os.path.expanduser("~/gpm_storm/data/largest_patch_statistics.parquet")  # f"feature_{granule_id}.parquet"


def preprocess_data(df):
    """
    Clean and standardize the dataset for K-Means.
    """
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


def find_optimal_clusters(data, min_k=2, max_k=30):
    """
    Find the optimal number of clusters using silhouette score.
    """
    print("Finding the optimal number of clusters...")

    best_k = min_k
    best_score = -1
    silhouette_scores = []

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        # Compute silhouette score
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

        print(f"K={k}: Silhouette Score = {score:.4f}")

        # Update best_k if the score is better
        if score > best_score:
            best_k = k
            best_score = score

    print(f"\nOptimal number of clusters: K={best_k} with silhouette score {best_score:.4f}\n")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(min_k, max_k + 1), silhouette_scores, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.grid()
    plt.show()

    return best_k


def perform_kmeans(data, n_clusters):
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
            data[labels == i, 0],
            data[labels == i, 1],
            label=f"Cluster {i + 1}",
            alpha=0.5,
        )

    # Plot cluster centers
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="black",
        marker="x",
        s=100,
        label="Centroids",
    )

    plt.title("K-Means Clustering")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    df = pd.read_parquet(FILEPATH)
    df_scaled = preprocess_data(df)

    # Optimize K-Means by finding the best K
    optimal_k = find_optimal_clusters(df_scaled)

    # Perform K-Means with the best K
    kmeans, labels, cluster_centers = perform_kmeans(df_scaled, optimal_k)

    # Visualization
    plot_clusters(df_scaled, labels, cluster_centers, optimal_k)


if __name__ == "__main__":
    main()
