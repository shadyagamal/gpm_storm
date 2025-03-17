#!/usr/bin/env python3
"""

@author: shadya
"""
#%% Imports
import numpy as np  # noqa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

filepath = "~/gpm_storm/data/patch_statistics.parquet"  # f"feature_{granule_id}.parquet"
df = pd.read_parquet(filepath)

#%% Step 1: Data Preprocessing
# Assuming your data is in a 2D array or DataFrame with rows as samples and columns as variables.
df_cleaned = df.dropna(axis=1)  # Remove rows with missing values A CHANGER PAR AXIS=0
df_cleaned = df_cleaned.select_dtypes(include=[np.number])  # Keep only numerical columns

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)

# %% Step 2: Clustering
# Define the number of clusters (k)
k = 3

# Create a K-Means model
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(df_scaled)

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# %% Step 3: Plot Clusters
# Visualize the clustering results
plt.figure(figsize=(8, 6))

# Scatter plot of data points with different colors for each cluster
for i in range(k):
    plt.scatter(
        df_scaled[labels == i, 0], df_scaled[labels == i, 1], label=f"Cluster {i + 1}", alpha=0.5
    )

# Plot cluster centers
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1], c="black", marker="x", s=40, label="Centroids"
)

plt.title("K-Means Clustering")
plt.legend()
plt.grid()
plt.show()


#%% You can access the cluster assignments for each data point
labels = kmeans.labels_
# %%
