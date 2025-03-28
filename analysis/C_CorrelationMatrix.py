#!/usr/bin/env python3
"""
Compute and visualize the correlation matrix of features from patch statistics.

@author: shadya
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FILEPATH = os.path.expanduser("~/gpm_storm/data/largest_patch_statistics.parquet")  # f"feature_{granule_id}.parquet"


def plot_correlation_matrix(df):
    """
    Compute and plot the correlation matrix of features.
    """
    correlation_matrix = df.corr()

    # Set up the figure
    plt.figure(figsize=(12, 8))  # Increase figure size for better readability
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm")

    # Display the plot
    plt.title("Feature Correlation Matrix")
    plt.show()


def main():
    """
    Load patch statistics, compute correlation matrix, and plot the heatmap.
    """
    # Load the data
    df = pd.read_parquet(FILEPATH)

    # Compute the correlation matrix
    # Plot the heatmap
    plot_correlation_matrix(df)


if __name__ == "__main__":
    main()
