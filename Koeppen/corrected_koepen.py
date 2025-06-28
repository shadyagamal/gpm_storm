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

# === Full Köppen-Geiger class mapping ===
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

# === Climate class to group mapping ===
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

# === Plot full Köppen-Geiger map ===
def plot_full_kg(data):
    n_classes = len(kg_classes)
    cmap = ListedColormap(plt.cm.tab20b(np.linspace(0, 1, n_classes)))
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)

    plt.figure(figsize=(14, 7))
    im = plt.imshow(data, cmap=cmap, norm=norm)
    plt.title("Köppen-Geiger Climate Classification (CHELSA V2.1)")
    plt.axis('off')

    patches = [
        mpatches.Patch(color=class_color_map[abbr], label=f"{code:02d} {abbr} – {desc}")
        for i, (code, (abbr, desc)) in enumerate(kg_classes.items())
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0., fontsize=8, title="Climate Zones")
    plt.tight_layout()
    plt.show()

plot_full_kg(kg_data)

# === Plot simplified groups ===
def plot_grouped_kg(data):
    grouped_data = np.vectorize(lambda x: group_to_id[kg_class_to_group.get(x, 'E')])(data)

    cmap = ListedColormap(['green', 'sandybrown', 'yellowgreen', 'lightblue', 'white'])  # A, B, C, D, E
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

# === Assign KG class and group to point samples ===
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

som_name = "SOM_Pmean_>_1_with_random_init"  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
df_bmu = pd.read_parquet(bmu_dir)
df_bmu = assign_kg_class_and_group(df_bmu, kg_data, transform)

kg_dist = df_bmu.groupby(["row", "col"])["kg_group"].value_counts(normalize=True).unstack().fillna(0)
kg_dist.plot(kind="bar", stacked=True, figsize=(14, 6), colormap="Set2")
plt.title("Climate Group Distribution per BMU")
plt.ylabel("Proportion")
plt.xlabel("BMU")
plt.legend(title="KG Group")
plt.tight_layout()
plt.show()


kg_group_counts = (
    df_bmu.groupby(["row", "col"])["kg_group"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)

dominant_kg_group = (
    df_bmu.groupby(["row", "col"])["kg_group"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index(name="dominant_kg_group")
)

# You can also pivot this into a 2D grid if needed for plotting:
grid = dominant_kg_group.pivot(index="row", columns="col", values="dominant_kg_group")
