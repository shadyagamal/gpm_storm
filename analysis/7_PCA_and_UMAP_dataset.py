#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:14:52 2023

@author: comi
"""

#%% PCA and UMAP for dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd 
from sklearn.cluster import KMeans
import numpy as np

def _process_nan_values(df, threshold_percentage):
    # Calculate the percentage of NaN values for each column
    nan_percentages = (df.isnull().sum() / len(df)) * 100

    # Display NaN percentages
    print("NaN Percentages for Each Column:")
    print(nan_percentages)

    # Create a list of columns to drop based on the threshold percentage
    columns_to_drop = nan_percentages[nan_percentages > threshold_percentage].index.tolist()

    # Create a new DataFrame without columns containing NaN values above the threshold
    df_no_nan = df.drop(columns=columns_to_drop)

    return df_no_nan

# Step 1: Data Preprocessing
# Assuming your data is in a 2D array or DataFrame with rows as samples and columns as variables.
filepath = "~/gpm_storm/data/patch_statistics.parquet"  # f"feature_{granule_id}.parquet"
df = pd.read_parquet(filepath)
df_cleaned = df.select_dtypes(include=[np.number])  # Keep only numerical columns

# Standardize the data
scaler = StandardScaler()
df_scaled = df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns, index=df_cleaned.index)

# Instantiate the PCA model
pca = PCA()

df_scaled = _process_nan_values(df_scaled, 1)
df_scaled = df_scaled.iloc[:, :-7]
# Fit the PCA model on the scaled data
pca.fit(df_scaled)

# Plot the cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

desired_variance = 0.95
num_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= desired_variance) + 1
print(f"Number of components to retain {desired_variance * 100}% of variance: {num_components}")

# Instantiate the PCA model
pca = PCA(n_components=num_components)

# Fit the PCA model on the scaled data and transform the data
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=[f'PC{i}' for i in range(1, num_components + 1)])


# Calculate the correlation matrix
correlation_matrix = df_pca.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of Principal Components')
plt.show()


# Specify the number of clusters
num_clusters = 5

# Instantiate the KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model on the principal components
df_pca['Cluster'] = kmeans.fit_predict(df_pca)

# Display the cluster assignments
print(df_pca['Cluster'].value_counts())


# Merge the original dataset with the principal components and cluster assignments
df_with_clusters = pd.concat([df_scaled, df_pca['Cluster']], axis=1)

# Calculate mean values for each variable within each cluster
cluster_means = df_with_clusters.groupby('Cluster').mean()

# Display the mean values for each cluster
print(cluster_means)

# Assuming pca is your fitted PCA model
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, num_components + 1)], index=df_scaled.columns)

# Display the loadings
print(loadings)
# %%
