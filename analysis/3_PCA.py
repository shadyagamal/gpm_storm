#!/usr/bin/env python3
"""
Created on Tue Oct 31 15:17:32 2023

@author: shadya
"""
#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

filepath = "~/gpm_storm/data/patch_statistics.parquet"  # f"feature_{granule_id}.parquet"
df = pd.read_parquet(filepath)

#%% Step 1: Data Preprocessing
# Assuming your data is in a 2D array or DataFrame with rows as samples and columns as variables.
df_cleaned = df.dropna(axis=1)  # Remove rows with missing values A CHANGER PAR AXIS=0
df_cleaned = df_cleaned.select_dtypes(include=[np.number])  # Keep only numerical columns

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)  # 'your_data' should be replaced with your data

# %% Step 2: PCA Calculation
n_components = 7  # Choose the number of components you want to retain
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_scaled)

#%% Step 3: Variance Explained
explained_variance_ratio = pca.explained_variance_ratio_
# You can assess how much variance is explained by each PC and decide on the number to retain.

#%% Step 4: Interpret and Use the PCs
# You can access the loadings using 'pca.components_' and use the transformed data
# in 'principal_components' for further analysis or visualization.

# Create a scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)

# Add labels to some data points for illustration (you can customize this)
for i, (x, y) in enumerate(zip(principal_components[:, 0], principal_components[:, 1])):
    if i == 0:
        plt.text(x, y, "A", fontsize=12, ha="right", va="bottom")
    elif i == 1:
        plt.text(x, y, "B", fontsize=12, ha="left", va="top")
    # Add more annotations as needed

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - First Two Principal Components")

# Annotate clusters or outliers if identified
# You can add code to identify and annotate clusters or outliers based on your data.

plt.grid()
plt.show()

#%% Variance explained plot
explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio_cumsum,
    marker="o",
    linestyle="--",
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by Principal Components")
plt.grid()
plt.show()
