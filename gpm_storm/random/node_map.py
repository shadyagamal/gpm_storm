import itertools
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import xarray as xr
import glob
import gpm
from gpm.visualization import plot_cartopy_background  # type: ignore
from gpm_storm.som.experiments import load_som
from gpm_storm.som.io import create_som_df_array
import matplotlib.patches as patches
from collections import defaultdict
import seaborn as sns
from sklearn.feature_selection import f_regression
from kgcpy import *
import datetime
from matplotlib.colors import LogNorm, Normalize
from xhistogram.xarray import histogram
from tqdm import tqdm
import pickle 
from scipy.ndimage import label
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def plot_mean_variable_per_node(arr_df, save_dir, variable="P_mean", high_q=0.99, low_q=0.01, fig=True):
    mean_values = np.full((10, 10), np.nan)

    for row in range(10):
        for col in range(10):
            df_node = arr_df[row, col]
            if not df_node.empty:
                mean_val = df_node[variable].mean()
                mean_values[row, col] = mean_val
                
    valid_means = mean_values[~np.isnan(mean_values)]
    high_thresh = np.quantile(valid_means, high_q)
    low_thresh = np.quantile(valid_means, low_q)

    for row in range(10):
        for col in range(10):
            val = mean_values[row, col]
            if np.isnan(val):
                continue
            if val >= high_thresh:
                node_characteristics[(row, col)].append(f"high {variable}")
            if val <= low_thresh:
                node_characteristics[(row, col)].append(f"low {variable}")

som_name = "SOM_Pmean_>_1_with_random_init"  
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}")
kde_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/KDE")
res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/0_Results")
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"

os.makedirs(figs_dir, exist_ok=True)
os.makedirs(kde_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
    
    
# LOAD
with open(os.path.join(res_dir,"cfad_totals.pkl"), "rb") as f:
    cfad_totals = pickle.load(f)
with open(os.path.join(res_dir,"cfad_convs.pkl"), "rb") as f:
    cfad_convs = pickle.load(f)
with open(os.path.join(res_dir,"cfad_strats.pkl"), "rb") as f:
    cfad_strats = pickle.load(f)
with open(os.path.join(res_dir,"cfad_total_tops.pkl"), "rb") as f:
    cfad_total_tops = pickle.load(f)
with open(os.path.join(res_dir,"cfad_others.pkl"), "rb") as f:
    cfad_others = pickle.load(f)
with open(os.path.join(res_dir,"wcc_flagss.pkl"), "rb") as f:
    wcc_flags = pickle.load(f)   

# --- Load ---
df_bmu = pd.read_parquet(bmu_dir)
som = load_som(som_dir=som_dir, som_name=som_name)
n_rows = som._n_rows
n_cols = som._n_columns
arr_df, arr_ds = build_som_patch_array(df_bmu, som, zarr_directory)

node_characteristics = defaultdict(list)
for col in df_bmu.columns[:134]:
    plot_mean_variable_per_node(
        arr_df, save_dir=figs_dir, variable=col, fig=False)
    

high_symbol="↑"
low_symbol="↓"
title="SOM Node Characteristics"
save_path=None
grid_shape = (10, 10)
high_mask = np.zeros(grid_shape, dtype=int)
low_mask = np.zeros(grid_shape, dtype=int)

# Populate masks
grid_shape = (10, 10)
high_mask = np.zeros(grid_shape, dtype=int)
low_mask = np.zeros(grid_shape, dtype=int)

# Populate masks
for (row, col), characteristics in node_characteristics.items():
    if any("high" in s for s in characteristics):
        high_mask[row, col] = 1
    if any("low" in s for s in characteristics):
        low_mask[row, col] = 1

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 9.5)
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
ax.grid(which="minor", color="lightgray", linestyle="--", linewidth=0.5)
ax.set_title(title)
ax.invert_yaxis()

# Add symbols
for (row, col), characteristics in node_characteristics.items():
    if any("high" in s for s in characteristics):
        ax.text(col, row, high_symbol, fontsize=18, ha='center', va='center', fontweight='bold')
    elif any("low" in s for s in characteristics):
        ax.text(col, row, low_symbol, fontsize=18, ha='center', va='center', fontweight='bold')

def draw_contours(mask, color):
    labeled_array, num_features = label(mask)
    for region_id in range(1, num_features + 1):
        patches = []
        region = (labeled_array == region_id)
        for r in range(region.shape[0]):
            for c in range(region.shape[1]):
                if region[r, c]:
                    # Define cell corners clockwise
                    square = [
                        (c - 0.5, r - 0.5),
                        (c + 0.5, r - 0.5),
                        (c + 0.5, r + 0.5),
                        (c - 0.5, r + 0.5)
                    ]
                    patches.append(Polygon(square))
        patch_collection = PatchCollection(patches, facecolor='none', edgecolor=color, linewidth=2)
        ax.add_collection(patch_collection)

draw_contours(high_mask, color='red')
draw_contours(low_mask, color='blue')

if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()


from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

def plot_som_node_characteristics_colored_contours(
    node_characteristics,
    unique_variables=None,
    high_symbol="↑",
    low_symbol="↓",
    title="SOM Node Characteristics (Multi-Variable)",
    save_path=None
):
    grid_shape = (10, 10)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.invert_yaxis()

    # Identify unique variables if not provided
    if unique_variables is None:
        variable_set = set()
        for vals in node_characteristics.values():
            for tag in vals:
                _, var = tag.split(" ", 1)
                variable_set.add(var)
        unique_variables = sorted(variable_set)

    # Assign color to each variable
    cmap = get_cmap("tab10")  # or "tab20", or any other
    var_colors = {var: to_hex(cmap(i % cmap.N)) for i, var in enumerate(unique_variables)}

    # Plot symbols (optional: just mark presence of any flag)
    for (row, col), characteristics in node_characteristics.items():
        symbol = ""
        if any("high" in s for s in characteristics):
            symbol += high_symbol
        if any("low" in s for s in characteristics):
            symbol += low_symbol
        if symbol:
            ax.text(col, row, symbol, fontsize=14, ha='center', va='center', fontweight='bold')

    # For each (flag, variable), draw colored contours
    from scipy.ndimage import label
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from collections import defaultdict

    # Create masks per (flag, variable)
    masks = defaultdict(lambda: np.zeros(grid_shape, dtype=int))
    for (row, col), characteristics in node_characteristics.items():
        for tag in characteristics:
            flag, var = tag.split(" ", 1)
            masks[(flag, var)][row, col] = 1

    # Plot masks
    for (flag, var), mask in masks.items():
        color = var_colors[var]
        labeled_array, num_features = label(mask)
        for region_id in range(1, num_features + 1):
            patches = []
            region = (labeled_array == region_id)
            for r in range(region.shape[0]):
                for c in range(region.shape[1]):
                    if region[r, c]:
                        square = [
                            (c - 0.5, r - 0.5),
                            (c + 0.5, r - 0.5),
                            (c + 0.5, r + 0.5),
                            (c - 0.5, r + 0.5)
                        ]
                        patches.append(Polygon(square))
            patch_collection = PatchCollection(
                patches,
                facecolor='none',
                edgecolor=color,
                linewidth=2,
                label=f"{flag} {var}"
            )
            ax.add_collection(patch_collection)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # avoid duplicates
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
plot_som_node_characteristics_colored_contours(node_characteristics)
