#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and visualize the PCA from patch statistics.

@author: shadya
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILEPATH = os.path.expanduser("~/gpm_storm/data/patch_statistics.parquet")  # f"feature_{granule_id}.parquet"


def preprocess_data(df, nan_threshold=1):
    """
    Clean and standardize the dataset for PCA.
    Assuming your data is in a 2D array or DataFrame with rows as samples and columns as variables.
    """
    print("Preprocessing data...")

    # Drop non-numeric columns
    df_cleaned = df.select_dtypes(include=[np.number])

    # Drop columns with too many NaN values
    nan_percentages = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
    columns_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

    # Standardize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned),
        columns=df_cleaned.columns,
        index=df_cleaned.index
    )

    print(f"Data preprocessing complete! {len(columns_to_drop)} columns removed due to NaNs.\n")
    return df_scaled

def perform_pca(df_scaled, variance_threshold=0.95):
    """
    Perform PCA and determine the number of components to retain.
    """
    print("Performing PCA...")
    
    pca = PCA()
    pca.fit(df_scaled)
    principal_components = pca.fit_transform(df_scaled)

    # Compute cumulative explained variance
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Plot first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    for i, (x, y) in enumerate(zip(principal_components[:, 0], principal_components[:, 1])):
        if i < 3:  # Label only a few points
            plt.text(x, y, f"P{i}", fontsize=12, ha="right", va="bottom")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - First Two Principal Components")
    plt.grid()
    plt.show()

    # Plot variance explained
    plt.figure(figsize=(8, 6))
    plt.plot(explained_variance_ratio, marker="o", linestyle="--")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs. Number of Components")
    plt.grid(True)
    plt.show()

    # Determine optimal number of components
    num_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
    print(f"✅ Retaining {num_components} components to explain {variance_threshold * 100}% variance.\n")

    # Apply PCA with the selected number of components
    pca = PCA(n_components=num_components)
    df_pca = pd.DataFrame(
        pca.fit_transform(df_scaled),
        columns=[f"PC{i}" for i in range(1, num_components + 1)]
    )

    return pca, df_pca


def main():
    df = pd.read_parquet(FILEPATH)
    df_scaled = preprocess_data(df)
    pca, df_pca = perform_pca(df_scaled)

if __name__ == "__main__":
    main()