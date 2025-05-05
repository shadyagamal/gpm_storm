#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gpm # noqa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from matplotlib.patches import Circle
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

filepath = "/home/gamal/gpm_storm/data/merged_data_total_0_with_bmus_umap_kmeans.parquet" 
df = pd.read_parquet(filepath)
vars = df.columns[0:-16]
vars_df = df[vars]

# Visualisation ---------------------------------------------------------------
# Variables frequency 
stats = ["P_mean", "P_std", "P_center_count", "P_sum",
         "P_max", "P_count", "MP_sum", "MP_contrib"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns
axes = axes.flatten()
for i, stat in enumerate(stats):
    sns.histplot(df[stat], bins=100, ax=axes[i], stat="probability")
    axes[i].set_title(stat)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Relative Frequency")
    axes[i].set_yscale("log")
plt.tight_layout()
plt.show()

# Boxplots
thresholds = [0, 1, 2, 5, 10, 20, 50, 80, 120]
variables = ["P_GT{t}_regions", "P_GT{t}_count", "P_GT{t}_mean", "P_GT{t}_sum",
             "P_GT{t}_min", "MA_LP_GT_{t}", "MiA_LP_GT_{t}", "AR_LP_GT_{t}"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten() 
for i, var_template in enumerate(variables):
    ax = axes[i]
    boxplot_data = []
    labels = []
    for t in thresholds:
        var_name = var_template.format(t=t)
        if var_name in df.columns:
            data = df[var_name].dropna()
            boxplot_data.append(data)
            labels.append(f"GT{t}") 
    sns.boxplot(data=boxplot_data, ax=ax, notch=True)
    ax.set_title(variables[i]) 
    # ax.set_yscale("log")  
    ax.set_xticklabels(labels, rotation=45)  
plt.tight_layout()
plt.show()

# Heatmap 
correlation_matrix = vars_df.corr()
plt.figure(figsize=(12, 8))  
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()


# PCA -------------------------------------------------------------------------
# Cleaning
df_cleaned = vars_df.select_dtypes(include=[np.number])
df_cleaned_cols = df_cleaned.dropna(axis=1)  
df_cleaned_rows = df_cleaned.dropna(axis=0)

# Standardization
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_cleaned),
    columns=df_cleaned.columns,
    index=df_cleaned.index)

df_scaled_cols = pd.DataFrame(
    scaler.fit_transform(df_cleaned_cols),
    columns=df_cleaned_cols.columns,
    index=df_cleaned_cols.index)

df_scaled_rows = pd.DataFrame(
    scaler.fit_transform(df_cleaned_rows),
    columns=df_cleaned_rows.columns,
    index=df_cleaned_rows.index)

# Fit
pca_cols = PCA().fit(df_scaled_cols)
pca_rows = PCA().fit(df_scaled_rows)

pc_cols = pca_cols.transform(df_scaled_cols)
pc_rows = pca_rows.transform(df_scaled_rows)

# Variance Explained
explained_var_cols = np.cumsum(pca_cols.explained_variance_ratio_)
explained_var_rows = np.cumsum(pca_rows.explained_variance_ratio_)

top_n = 5 
feature_importance_cols = np.abs(pca_cols.components_[0]) + np.abs(pca_cols.components_[1])
feature_importance_rows = np.abs(pca_rows.components_[0]) + np.abs(pca_rows.components_[1])

top_indices_cols = np.argsort(feature_importance_cols)[-top_n:]
top_indices_rows = np.argsort(feature_importance_rows)[-top_n:]

num_components_cols = np.argmax(explained_var_cols >= 0.95) + 1
num_components_rows = np.argmax(explained_var_rows >= 0.95) + 1

pca_cols = PCA(n_components=num_components_cols).fit(df_scaled_cols)
df_pca_cols = pd.DataFrame(
    pca_cols.transform(df_scaled_cols),
    columns=[f"PC{i}" for i in range(1, num_components_cols + 1)]
)

pca_rows = PCA(n_components=num_components_rows).fit(df_scaled_rows)
df_pca_rows = pd.DataFrame(
    pca_rows.transform(df_scaled_rows),
    columns=[f"PC{i}" for i in range(1, num_components_rows + 1)]
)
print(f"Retaining {num_components_cols} components (Dropped Columns) to explain 95% variance.")
print(f"Retaining {num_components_rows} components (Dropped Rows) to explain 95% variance.")

# Scatter Plot - Dropped Columns - Dropped Rows
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(pc_cols[:, 0], pc_cols[:, 1], alpha=0.5, label="Samples")
loadings_cols = pca_cols.components_[:2, :]
for i in top_indices_cols:
    axes[0].arrow(0, 0, loadings_cols[0, i] * 3, loadings_cols[1, i] * 3, alpha=0.7, head_width=0.05)
    axes[0].text(loadings_cols[0, i] * 3, loadings_cols[1, i] * 3, df_scaled_cols.columns[i], fontsize=10)
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].set_title("PCA (Dropped Columns)")
axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.5)
axes[0].axvline(0, color="grey", linestyle="--", linewidth=0.5)
axes[0].grid()

axes[1].scatter(pc_rows[:, 0], pc_rows[:, 1], alpha=0.5, label="Samples")
loadings_rows = pca_rows.components_[:2, :]
for i in top_indices_rows:
    axes[1].arrow(0, 0, loadings_rows[0, i] * 3, loadings_rows[1, i] * 3, alpha=0.7, head_width=0.05)
    axes[1].text(loadings_rows[0, i] * 3, loadings_rows[1, i] * 3, df_scaled_rows.columns[i], fontsize=10)
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_title("PCA (Dropped Rows)")
axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.5)
axes[1].axvline(0, color="grey", linestyle="--", linewidth=0.5)
axes[1].grid()

plt.tight_layout()
plt.show()

# Variance Explained Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var_cols) + 1), explained_var_cols, marker="o", label="Columns")
plt.plot(range(1, len(explained_var_rows) + 1), explained_var_rows, marker="s", label="Rows")
plt.axhline(y=0.95, linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
plt.legend()
plt.grid()
plt.show()

top_n = 10  
feature_importance_cols = np.abs(pca_cols.components_[0]) + np.abs(pca_cols.components_[1])
top_indices_cols = np.argsort(feature_importance_cols)[-top_n:]

# PCA Correlation Circle
ccircle = []
eucl_dist = []
for col in df_scaled_cols.columns:  
    corr1 = np.corrcoef(df_scaled_cols[col], pc_cols[:, 0])[0, 1]
    corr2 = np.corrcoef(df_scaled_cols[col], pc_cols[:, 1])[0, 1]
    ccircle.append((corr1, corr2))
    eucl_dist.append(np.sqrt(corr1**2 + corr2**2))
    
ccircle = np.array(ccircle)
top_indices = np.argsort(eucl_dist)[-top_n:]

# Plot correlation circle
with plt.style.context(('seaborn-v0_8-whitegrid')):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (corr1, corr2) in enumerate(ccircle):
        arrow_col = plt.cm.cividis((eucl_dist[i] - np.min(eucl_dist)) / (np.max(eucl_dist) - np.min(eucl_dist)))
        ax.arrow(0, 0, corr1, corr2, lw=2, color=arrow_col, head_width=0.05, head_length=0.05)
        ax.text(corr1 * 1.1, corr2 * 1.1, df_scaled_cols.columns[i], fontsize=10)
    circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='gray', lw=1, linestyle="--")
    ax.axvline(0, color='gray', lw=1, linestyle="--")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA Correlation Circle")
plt.tight_layout()
plt.show()

# Plot correlation circle with less variables
fig, ax = plt.subplots(figsize=(6, 6))
for i in top_indices:
    x, y = ccircle[i]
    ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, alpha=0.75)
    ax.text(x * 1.1, y * 1.1, df_cleaned_cols.columns[i], fontsize=10)

# Draw unit circle
circle = Circle((0, 0), 1, facecolor="none", edgecolor="k", linewidth=1, alpha=0.5)
ax.add_patch(circle)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("PCA Correlation Circle (Top 10 Variables)")
plt.grid()
plt.show()


# UMAP
# reducer = umap.UMAP()
# joblib.dump(reducer, "umap_model.joblib")

reducer = joblib.load("/home/gamal/gpm_storm/data/umap_model.joblib")
embedding = reducer.fit_transform(df_scaled_cols)

plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection")
plt.grid(True)
plt.show()

# Color by Key Features
# for var in vars:
#     plt.figure(figsize=(8, 6))
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=df[var], cmap="viridis", alpha=0.5)
#     plt.colorbar(label=var)
#     plt.xlabel("UMAP 1")
#     plt.ylabel("UMAP 2")
#     plt.title(f"UMAP Projection Colored by {var}")
#     plt.grid(True)
#     plt.show()

# Feature Correlation in UMAP Space
# umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
# umap_df.to_parquet("umap_embedding.parquet")
umap_df = pd.read_parquet("/home/gamal/gpm_storm/data/umap_embedding.parquet")
embedding = umap_df
corr_matrix = pd.concat([umap_df, df_scaled_cols], axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.iloc[:2, 2:], cmap="coolwarm", annot=True)
plt.title("Correlation of UMAP with Original Features")
plt.show()

# Kmeans
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
score = silhouette_score(embedding, kmeans.labels_)
print(score)

# df["cluster"] = kmeans.labels_
umap_df["kmeans_cluster"] = kmeans.labels_


plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=df["cluster"], cmap="tab10", alpha=0.5)
plt.colorbar(label="Cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP with K-Means Clustering")
plt.grid(True)
plt.show()

# # Save updated DataFrame with UMAP and KMEANS
# import os
# df["UMAP0"] = umap_df.iloc[:, 0]
# df["UMAP1"] = umap_df.iloc[:, 1]
# df["kmeans_cluster"] = kmeans.labels_
# new_filepath = os.path.expanduser("~/gpm_storm/data/merged_data_total_0_with_bmus_umap_kmeans.parquet")
# df.to_parquet(new_filepath)


# Binned maps
lat_bins = np.arange(-90, 91, 1)  
lon_bins = np.arange(-180, 181, 1)  
df["lat_bin"] = np.digitize(df["lat"], bins=lat_bins) - 1
df["lon_bin"] = np.digitize(df["lon"], bins=lon_bins) - 1
agg_df = df.groupby(["lat_bin", "lon_bin"]).mean(numeric_only=True).reset_index()
agg_df["lat"] = lat_bins[agg_df["lat_bin"]]
agg_df["lon"] = lon_bins[agg_df["lon_bin"]]

for var in variables[:-8]:
    heatmap = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)  # Initialize with NaNs
    for _, row in agg_df.iterrows():
        lat_idx = np.searchsorted(lat_bins, row["lat"], side="right") - 1
        lon_idx = np.searchsorted(lon_bins, row["lon"], side="right") - 1
        if 0 <= lat_idx < heatmap.shape[0] and 0 <= lon_idx < heatmap.shape[1]:
            heatmap[lat_idx, lon_idx] = row[var]

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    mesh = ax.pcolormesh(lon_bins, lat_bins, heatmap, cmap="viridis", transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical")
    cbar.set_label(f"{var} ")
    plt.title(f"Global Distribution of {var} (2°x2° bins)")
    plt.show()