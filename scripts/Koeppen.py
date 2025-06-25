#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:00:11 2025

@author: gamal
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:00:11 2025
@author: gamal
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

# Load image
img = Image.open('/home/gamal/lorenzo_gpm_storm/figs/koppen.png').convert('RGB')
img_array = np.array(img)
height, width = img_array.shape[:2]

# Define Koppen colors and categories
koppen_colors = {
    'Af': (106, 199, 183),
    'Aw': (198, 232, 226),
    'BS': (237, 93, 33),
    'BW': (239, 195, 34),
    'Cs': (104, 165, 69),
    'Cw': (169, 181, 61),
    'Cf': (50, 149, 70),
    # 'Dw': (93, 124, 137),
    'Df': (62, 87, 97),
    'ET': (122, 135, 147),
    'EF': (160, 172, 180),
}

categories = list(koppen_colors.keys())
color_centers = np.array([koppen_colors[cat] for cat in categories])

# Reshape image to (num_pixels, 3)
pixels = img_array.reshape(-1, 3)

# Initialize KMeans with Koppen colors as initial centers
kmeans = KMeans(n_clusters=len(categories), init=color_centers, n_init=1, random_state=42)
kmeans.fit(pixels)

# Assign each pixel to a cluster
labels = kmeans.labels_
classified_map = labels.reshape(height, width)

# Create colormap for visualization
colors = color_centers / 255
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(categories)), ncolors=len(categories))

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
extent = [-180, 180, -90, 90]
im = ax.imshow(classified_map, cmap=cmap, norm=norm, extent=extent, origin='upper')
cbar = fig.colorbar(im, ticks=np.arange(len(categories)))
cbar.ax.set_yticklabels(categories)
cbar.set_label('Koppen Climate Zones (KMeans Quantization)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Koppen Climate Zones Classified via KMeans')
plt.show()


structure = generate_binary_structure(2, 1)  # 4-connectivity

def smooth_cluster_map(class_map, cluster_idx, structure):
    mask = (class_map == cluster_idx)
    opened = binary_opening(mask, structure=structure, iterations=2)
    closed = binary_closing(opened, structure=structure, iterations=3)
    return closed

smoothed_map = np.full(classified_map.shape, -1, dtype=int)

for cluster_idx in range(len(categories)):
    smoothed_mask = smooth_cluster_map(classified_map, cluster_idx, structure)
    smoothed_map[smoothed_mask] = cluster_idx

# Now plot the smoothed map:

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = np.array([koppen_colors[cat] for cat in categories]) / 255
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(categories)), ncolors=len(categories))

fig, ax = plt.subplots(figsize=(12, 6))
extent = [-180, 180, -90, 90]
im = ax.imshow(smoothed_map, cmap=cmap, norm=norm, extent=extent, origin='upper')
cbar = fig.colorbar(im, ticks=np.arange(len(categories)))
cbar.ax.set_yticklabels(categories)
cbar.set_label('Koppen Climate Zones (Smoothed)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Smoothed Koppen Climate Zones after KMeans')
plt.show()



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from scipy.stats import mode

# Load image
img = Image.open('/home/gamal/lorenzo_gpm_storm/figs/koppen.png').convert('RGB')
img_array = np.array(img)
height, width = img_array.shape[:2]

# Define Koppen colors and categories
koppen_colors = {
    'Af': (106, 199, 183),
    'Aw': (198, 232, 226),
    'BS': (237, 93, 33),
    'BW': (239, 195, 34),
    'Cs': (104, 165, 69),
    'Cw': (169, 181, 61),
    'Cf': (50, 149, 70),
    # 'Dw': (93, 124, 137),
    'Df': (62, 87, 97),
    'ET': (122, 135, 147),
    'EF': (160, 172, 180),
}

categories = list(koppen_colors.keys())
color_centers = np.array([koppen_colors[cat] for cat in categories])

# Reshape image to (num_pixels, 3)
pixels = img_array.reshape(-1, 3)

# Initialize KMeans with Koppen colors as initial centers
kmeans = KMeans(n_clusters=len(categories), init=color_centers, n_init=1, random_state=42)
kmeans.fit(pixels)

# Assign each pixel to a cluster
labels = kmeans.labels_
classified_map = labels.reshape(height, width)

# Define lat/lon per pixel based on extent
lon = np.linspace(-180, 180, width, endpoint=False)  # longitudes
lat = np.linspace(90, -90, height, endpoint=False)   # latitudes decreasing (top to bottom)

# Create 2D grids for lon/lat per pixel
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Quantize lat/lon to integer degrees: lat bins 0 to 179, lon bins 0 to 359
lat_bins = np.floor(90 - lat_grid).astype(int)  # maps 90 to 0 and -90 to 179
lon_bins = np.floor(lon_grid + 180).astype(int) # maps -180 to 0 and 179 to 359

# Prepare empty quantized map
quantized_map = np.full((180, 360), -1, dtype=int)

# Aggregate by majority vote in each 1x1 deg cell
for i in range(180):
    for j in range(360):
        mask = (lat_bins == i) & (lon_bins == j)
        if np.any(mask):
            # majority vote of labels in this bin
            bin_labels = classified_map[mask]
            quantized_map[i, j] = mode(bin_labels, axis=None).mode[0]

# Now apply smoothing on quantized_map
structure = generate_binary_structure(2, 1)  # 4-connectivity

def smooth_cluster_map(class_map, cluster_idx, structure):
    mask = (class_map == cluster_idx)
    opened = binary_opening(mask, structure=structure, iterations=2)
    closed = binary_closing(opened, structure=structure, iterations=3)
    return closed

smoothed_map = np.full(quantized_map.shape, -1, dtype=int)

for cluster_idx in range(len(categories)):
    smoothed_mask = smooth_cluster_map(quantized_map, cluster_idx, structure)
    smoothed_map[smoothed_mask] = cluster_idx

# Plotting quantized + smoothed map
colors = np.array([koppen_colors[cat] for cat in categories]) / 255
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(categories)), ncolors=len(categories))

fig, ax = plt.subplots(figsize=(12, 6))
extent = [-180, 180, -90, 90]
im = ax.imshow(smoothed_map, cmap=cmap, norm=norm, extent=extent, origin='upper', interpolation='nearest')
cbar = fig.colorbar(im, ticks=np.arange(len(categories)))
cbar.ax.set_yticklabels(categories)
cbar.set_label('Koppen Climate Zones (1°x1° Quantized & Smoothed)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Koppen Climate Zones Quantized at 1° x 1° Resolution')
plt.show()
