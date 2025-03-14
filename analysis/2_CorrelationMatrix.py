#!/usr/bin/env python3
"""
Created on Tue Oct 31 15:16:52 2023

@author: Shadya
"""

# %% correlation matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filepath = "~/gpm_storm/data/patch_statistics.parquet"  # f"feature_{granule_id}.parquet"

df = pd.read_parquet(filepath)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a larger heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")

# Display the plot
plt.title("Correlation Matrix")
plt.show()