#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:52:50 2025

@author: gamal
"""

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

custom_colors = {
    "NA": "#ffffff",  
    "EF": "#6395fe",
    "ET": "#63fcfc",
    "Dwd": "#6a23ae",
    "Dwc": "#8858b0",
    "Dwb": "#987cb1",
    "Dwa": "#cab5fe",
    "Dsd": "#c8c9c9",
    "Dsc": "#e3c6f8",
    "Dsb": "#fbb4fc",
    "Dsa": "#fa6bfa",
    "Dfd": "#c81488",
    "Dfc": "#c800c8",
    "Dfb": "#630163",
    "Dfa": "#320232",
    "Cwd": "#5c3f02",
    "Cwc": "#996633",
    "Cwb": "#956603",
    "Cwa": "#b46400",
    "Csc": "#c9fd00",
    "Csb": "#93fc00",
    "Csa": "#00fa00",
    "Cfc": "#007700",
    "Cfb": "#014e01",
    "Cfa": "#002f00",
    "BSk": "#cca753",
    "BSh": "#cc8b13",
    "BWk": "#ffff65",
    "BWh": "#fece00",
    "Aw": "#fdcdcd",
    "As": "#fa9797",
    "Am": "#fb0000",
    "Af": "#920202"
}
# Merge KG class names
kg_code_to_abbr = {k: abbr for k, (abbr, _) in kg_classes.items()}
df_bmu["kg_abbr"] = df_bmu["kg_class"].map(kg_code_to_abbr)

# Filter valid data
valid_df = df_bmu.dropna(subset=["kg_group", "kg_abbr", "row", "col"])


unique_nodes = (
    valid_df[["row", "col"]]
    .drop_duplicates()
    .values
    .tolist()
)
max_row = valid_df["row"].max()
unique_nodes = sorted(unique_nodes, key=lambda x: (x[0], x[1]))

n_nodes = len(unique_nodes)
n_cols = 10
n_rows = int(np.ceil(n_nodes / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

# Custom color palette for up to 31 classes
all_classes = [abbr for _, (abbr, _) in sorted(kg_classes.items())]
class_color_map = {abbr: custom_colors.get(abbr, "#000000") for abbr in all_classes}

for idx, (row_val, col_val) in enumerate(unique_nodes):
    ax = axes[idx // n_cols][idx % n_cols]
    node_data = valid_df[(valid_df["row"] == row_val) & (valid_df["col"] == col_val)]

    if node_data.empty:
        ax.axis("off")
        continue

    # Group counts: (group, class) -> count
    group_class_counts = (
        node_data.groupby(["kg_group", "kg_abbr"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to total samples in the node
    total_samples = group_class_counts.values.sum()
    group_class_props = group_class_counts / total_samples  # --> Each bar height is a fraction

    # Plot stacked bar per group (A–E), stacked by class
    bottom = np.zeros(len(group_class_props))
    x = np.arange(len(group_class_props))
    for cls in all_classes:
        if cls in group_class_props.columns:
            y = group_class_props[cls].values
            ax.bar(x, y, bottom=bottom, color=class_color_map[cls], label=cls)
            bottom += y

    ax.set_xticks(x)
    ax.set_xticklabels(group_class_props.index)
    ax.set_ylim(0, 1)
    ax.set_title(f"SOM Node ({row_val}, {col_val})", fontsize=10)
    ax.tick_params(axis='x', rotation=45)

# Global legend
handles = [
    mpatches.Patch(color=class_color_map[cls], label=cls)
    for cls in all_classes if any(valid_df["kg_abbr"] == cls)
]
fig.legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc="center left", title="KG Class")
plt.tight_layout()
plt.show()





from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_full_kg(data):
    # Map integer codes to abbreviations in the correct order
    ordered_abbrs = [abbr for _, (abbr, _) in sorted(kg_classes.items())]

    # Build color list in the order of codes
    color_list = [class_color_map.get(abbr, "#000000") for abbr in ordered_abbrs]

    # Custom colormap and normalization
    cmap = ListedColormap(color_list)
    n_classes = len(ordered_abbrs)
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)

    plt.figure(figsize=(14, 7))
    im = plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Köppen-Geiger Climate Classification (CHELSA V2.1)")
    plt.axis('off')

    # Legend patches
    patches = [
        mpatches.Patch(color=class_color_map[abbr], label=f"{code:02d} {abbr} – {desc}")
        for code, (abbr, desc) in sorted(kg_classes.items())
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=8, title="Climate Zones")
    plt.tight_layout()
    plt.show()
