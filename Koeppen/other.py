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
    0: ('Af', 'equatorial fully humid'),    1: ('Am', 'equatorial monsoonal'),
    2: ('As', 'equatorial summer dry'),     3: ('Aw', 'equatorial winter dry'),
    4: ('BWk', 'cold desert'),              5: ('BWh', 'hot desert'),
    6: ('BSk', 'cold steppe'),              7: ('BSh', 'hot steppe'),
    8: ('Cfa', 'humid subtropical'),        9: ('Cfb', 'marine west coast'),
    10: ('Cfc', 'subpolar oceanic'),        11: ('Csa', 'hot-summer Mediterranean'),
    12: ('Csb', 'warm-summer Mediterranean'), 13: ('Csc', 'cold-summer Mediterranean'),
    14: ('Cwa', 'dry-winter humid subtropical'), 15: ('Cwb', 'dry-winter subtropical highland'),
    16: ('Cwc', 'subtropical highland'),    17: ('Dfa', 'humid continental, hot summer'),
    18: ('Dfb', 'humid continental, warm summer'), 19: ('Dfc', 'subarctic'),
    20: ('Dfd', 'extremely continental subarctic'), 21: ('Dsa', 'Mediterranean, hot summer'),
    22: ('Dsb', 'Mediterranean, warm summer'), 23: ('Dsc', 'Mediterranean, cold summer'),
    24: ('Dsd', 'Mediterranean, extremely continental'), 25: ('Dwa', 'dry-winter, hot summer'),
    26: ('Dwb', 'dry-winter, warm summer'), 27: ('Dwc', 'dry-winter, cold summer'),
    28: ('Dwd', 'dry-winter, extremely continental'), 29: ('ET', 'tundra'),
    30: ('EF', 'frost')
}
kg_code_to_abbr = {k: abbr for k, (abbr, _) in kg_classes.items()}

# === Grouping of classes ===
kg_class_to_group = {
    **{k: 'A' for k in [0, 1, 2, 3]},
    **{k: 'B' for k in [4, 5, 6, 7]},
    **{k: 'C' for k in range(8, 17)},
    **{k: 'D' for k in range(17, 29)},
    29: 'E', 30: 'E'
}
group_names = {
    'A': 'Tropical', 'B': 'Arid', 'C': 'Temperate',
    'D': 'Cold', 'E': 'Polar'
}
group_to_id = {g: i for i, g in enumerate(group_names)}
id_to_group = {v: k for k, v in group_to_id.items()}

# === Custom color mapping ===
# custom_colors = {
#     "NA": "#ffffff",  
#     "EF": "#6395fe", "ET": "#63fcfc",
#     "Dwd": "#6a23ae", "Dwc": "#8858b0", "Dwb": "#987cb1", "Dwa": "#cab5fe",
#     "Dsd": "#c8c9c9", "Dsc": "#e3c6f8", "Dsb": "#fbb4fc", "Dsa": "#fa6bfa",
#     "Dfd": "#c81488", "Dfc": "#c800c8", "Dfb": "#630163", "Dfa": "#320232",
#     "Cwc": "#996633", "Cwb": "#956603", "Cwa": "#b46400",
#     "Csc": "#c9fd00", "Csb": "#93fc00", "Csa": "#00fa00",
#     "Cfc": "#007700", "Cfb": "#014e01", "Cfa": "#002f00",
#     "BSk": "#cca753", "BSh": "#cc8b13", "BWk": "#ffff65", "BWh": "#fece00",
#     "Aw": "#fdcdcd", "As": "#fa9797", "Am": "#fb0000", "Af": "#920202"
# }
s

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
