import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from rasterio.transform import rowcol
import os
import pandas as pd

# === Load and preprocess data ===
kg_filepath = "/ltenas2/data/GPM_STORM_DB/keop/CHELSA_kg0_1981-2010_V.2.1.tif"

with rasterio.open(kg_filepath) as src:
    kg_data = src.read(1)
    kg_nodata = src.nodata
    transform = src.transform

if kg_nodata is not None:
    kg_data = np.ma.masked_where(kg_data == kg_nodata, kg_data)

# === Köppen-Geiger class mapping ===
kg_classes = {
    1: ('Af', 'equatorial fully humid'),    2: ('Am', 'equatorial monsoonal'),
    3: ('As', 'equatorial summer dry'),     4: ('Aw', 'equatorial winter dry'),
    5: ('BWk', 'cold desert'),              6: ('BWh', 'hot desert'),
    7: ('BSk', 'cold steppe'),              8: ('BSh', 'hot steppe'),
    9: ('Cfa', 'humid subtropical'),        10: ('Cfb', 'marine west coast'),
    11: ('Cfc', 'subpolar oceanic'),        12: ('Csa', 'hot-summer Mediterranean'),
    13: ('Csb', 'warm-summer Mediterranean'), 14: ('Csc', 'cold-summer Mediterranean'),
    15: ('Cwa', 'dry-winter humid subtropical'), 16: ('Cwb', 'dry-winter subtropical highland'),
    17: ('Cwc', 'subtropical highland'),    18: ('Dfa', 'humid continental, hot summer'),
    19: ('Dfb', 'humid continental, warm summer'), 20: ('Dfc', 'subarctic'),
    21: ('Dfd', 'extremely continental subarctic'), 22: ('Dsa', 'Mediterranean, hot summer'),
    23: ('Dsb', 'Mediterranean, warm summer'), 24: ('Dsc', 'Mediterranean, cold summer'),
    25: ('Dsd', 'Mediterranean, extremely continental'), 26: ('Dwa', 'dry-winter, hot summer'),
    27: ('Dwb', 'dry-winter, warm summer'), 28: ('Dwc', 'dry-winter, cold summer'),
    29: ('Dwd', 'dry-winter, extremely continental'), 30: ('ET', 'tundra'),
    31: ('EF', 'frost')
}
kg_code_to_abbr = {k: abbr for k, (abbr, _) in kg_classes.items()}

# === Grouping of classes ===
kg_class_to_group = {
    **{k: 'A' for k in [1, 2, 3,4]},
    **{k: 'B' for k in [5, 6, 7,8]},
    **{k: 'C' for k in range(9, 18)},
    **{k: 'D' for k in range(18, 30)},
    30: 'E', 31: 'E'
}
group_names = {
    'A': 'Tropical', 'B': 'Arid', 'C': 'Temperate',
    'D': 'Cold', 'E': 'Polar'
}
group_to_id = {g: i for i, g in enumerate(group_names)}
id_to_group = {v: k for k, v in group_to_id.items()}

# === Custom color mapping ===
custom_colors = {
    "NA": "#ffffff",  # No data, white

    # Polar climates (E) - Blues
    "EF": "#6395fe",  # dark blue
    "ET": "#63fcfc",  # sky blue

    # Dry climates (B) - Sandy/Desert tones (orange to brown)
    "BSk": "#cca753",
    "BSh": "#cc8b13",
    "BWk": "#ffff65",
    "BWh": "#fece00",

    # Temperate climates (C) 
    # Mediterranean - Green
    "Csa": "#f57c00",  # deep orange
    "Csb": "#fb8c00",  # vivid orange
    "Csc": "#ffb74d",  # light orange

    # Marine/subtropical - Yellow-green
    "Cfa": "#007f7f",  # fresh green
    "Cfb": "#339f9f",  # mid green
    "Cfc": "#99d9d9",  # forest green

    # Highland/dry-winter - Browns
    "Cwa": "#c19a6b",  # light brown
    "Cwb": "#9e7951",  # chestnut
    "Cwc": "#7a5b3a",  # deep brown

    # Continental climates (D)
    # Humid continental - Magentas
    "Dfa": "#e57eb4",  # magenta
    "Dfb": "#cf5a9b",  # deep pink
    "Dfc": "#b03c83",  # violet
    "Dfd": "#84235e",  # plum

    # Dry-winter - Purple
    "Dwa": "#b39ddb",  # lavender
    "Dwb": "#9575cd",  # soft purple
    "Dwc": "#7e57c2",  # mid purple
    "Dwd": "#5e35b1",  # royal purple

    # Mediterranean continental - Pinks
    "Dsa": "#f4a6b6",  # soft pink
    "Dsb": "#ed6d92",  # medium pink
    "Dsc": "#c7446c",  # deep pink
    "Dsd": "#a1274d",  # burgundy

    # Tropical climates (A) - Teals and Greens
   "Af": "#33691e",  # Tropical rainforest — deep forest green
    "Am": "#689f38",  # Tropical monsoon — rich mid green
    "Aw": "#8bc34a",  # Tropical savanna — strong deep green
    "As": "#b6f118",  # Tropical dry savanna — lighter green-teal for contrast
}

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

n_classes = len(kg_classes)
colors_list = []
for i in range(1, n_classes + 1):
    abbr = kg_classes[i][0]
    color = custom_colors.get(abbr)
    if color is None:
        print(f"Missing color for {abbr} at index {i}")
        color = "#ffffff"
    colors_list.append(color)

cmap = ListedColormap(colors_list)
norm = BoundaryNorm(np.arange(0.5, n_classes + 1.5), n_classes)

# Main figure without legend
plt.figure(figsize=(14, 7))
im = plt.imshow(kg_data, cmap=cmap, norm=norm)
plt.title("Köppen-Geiger Climate Classification (CHELSA V2.1)")
plt.axis('off')
plt.tight_layout()
plt.show()

# Separate legend figure with 2 columns
patches = [
    mpatches.Patch(color=custom_colors[abbr], label=f"{code:02d} {abbr} – {desc}")
    for code, (abbr, desc) in kg_classes.items()
]

fig_legend = plt.figure(figsize=(6, len(patches)//2 * 0.5))  # height depends on number of patches
plt.legend(handles=patches, loc='center', fontsize=8, title="Climate Zones", ncol=2)
plt.axis('off')
plt.tight_layout()
plt.show()

# === Plot full Köppen-Geiger map ===
def plot_full_kg(data):
    n_classes = len(kg_classes)
    cmap = ListedColormap([custom_colors.get(kg_classes[i][0], "#ffffff") for i in range(1,n_classes+1)])
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)

    plt.figure(figsize=(14, 7))
    im = plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Köppen-Geiger Climate Classification (CHELSA V2.1)")
    plt.axis('off')

    patches = [
        mpatches.Patch(color=custom_colors[abbr], label=f"{code:02d} {abbr} – {desc}")
        for code, (abbr, desc) in kg_classes.items()
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=8, title="Climate Zones")
    plt.tight_layout()
    plt.show()

plot_full_kg(kg_data)

# === Simplified Köppen groups map ===
def plot_grouped_kg(data):
    grouped_data = np.vectorize(lambda x: group_to_id[kg_class_to_group.get(x, 'E')])(data)
    cmap = ListedColormap(['green', 'sandybrown', 'yellowgreen', 'lightblue', 'white'])
    norm = BoundaryNorm(np.arange(-0.5, len(group_names)+0.5), len(group_names))

    plt.figure(figsize=(12, 6))
    plt.imshow(grouped_data, cmap=cmap, norm=norm)
    plt.title("Simplified Köppen-Geiger Climate Groups")
    plt.axis('off')

    legend_patches = [
        mpatches.Patch(color=cmap(i), label=f"{id_to_group[i]} – {group_names[id_to_group[i]]}")
        for i in range(len(group_names))
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Climate Group")
    plt.tight_layout()
    plt.show()

plot_grouped_kg(kg_data)

# === Assign Köppen class and group to samples ===
def assign_kg_class_and_group(df, kg_data, transform):
    full_classes = []
    group_classes = []

    for lat, lon in zip(df['lat'], df['lon']):
        try:
            row, col = rowcol(transform, lon, lat)
            kg_class = kg_data[row, col]
            full_classes.append(kg_class)
            group_classes.append(kg_class_to_group.get(kg_class, 'Unknown'))
        except Exception:
            full_classes.append(None)
            group_classes.append(None)

    df["kg_class"] = full_classes
    df["kg_group"] = group_classes
    return df

# === Load BMU dataframe and assign KG data ===
som_name = "SOM_Pmean_>_1_with_random_init"
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
df_bmu = pd.read_parquet(bmu_dir)
df_bmu = assign_kg_class_and_group(df_bmu, kg_data, transform)
df_bmu["kg_abbr"] = df_bmu["kg_class"].map(kg_code_to_abbr)

# === Climate group distribution plot ===
kg_dist = df_bmu.groupby(["row", "col"])["kg_group"].value_counts(normalize=True).unstack().fillna(0)
kg_dist.plot(kind="bar", stacked=True, figsize=(14, 6), colormap="Set2")
plt.title("Climate Group Distribution per BMU")
plt.ylabel("Proportion")
plt.xlabel("BMU")
plt.legend(title="KG Group")
plt.tight_layout()
plt.show()

# === Dominant climate group per BMU ===
dominant_kg_group = (
    df_bmu.groupby(["row", "col"])["kg_group"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index(name="dominant_kg_group")
)

# Optional: Pivot to 2D grid for visualization
grid = dominant_kg_group.pivot(index="row", columns="col", values="dominant_kg_group")


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