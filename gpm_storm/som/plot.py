#!/usr/bin/env python3
"""
Created on Thu Dec 21 11:08:23 2023

@author: ghiggi
"""

import matplotlib.pyplot as plt
import numpy as np

# from gpm.utils.utils_cmap import get_colorbar_settings


def _remove_axis(ax):
    ax.set_title("")  # Set title to an empty string
    ax.set_xlabel("")  # Set xlabel to an empty string
    ax.set_ylabel("")  # Set ylabel to an empty string
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def _get_patch_image(img):
    max_value_position = np.unravel_index(np.argmax(img), img.shape)
    center_y, center_x = max_value_position
    if center_x < 25:
        img = img[:, 0:49]
    elif (img.shape[1] - center_x) > 25:
        start_x = center_x - 24
        end_x = center_x + 25
        img = img[:, start_x:end_x]
    else:
        img = img[:, -49:]
    return img


def plot_som_array_datasets(arr_ds, figsize=(10, 10), plot_kwargs={}, cbar_kwargs={}, variable="precipRateNearSurface"):
    # Retrieve SOM grid size
    nrows, ncols = arr_ds.shape
    # Set default plotting parameters if not provided
    if plot_kwargs is None:
        plot_kwargs = {"cmap": "viridis", "interpolation": "nearest"}
    if cbar_kwargs is None:
        cbar_kwargs = {"shrink": 0.8, "aspect": 20}

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            da = arr_ds[i, j][variable]
            img = _get_patch_image(da.data)
            ax.imshow(img, **plot_kwargs)
            _remove_axis(ax)
    return fig


def plot_images(list_ds, ncols=5, figsize=(15, 5), plot_kwargs={}, cbar_kwargs={}, variable="precipRateNearSurface"):

    num_images = len(list_ds)

    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(np.ceil(num_images / ncols))  # Adjust as needed
    num_cols = min(num_images, ncols)

    # Set default plotting parameters if not provided
    if plot_kwargs is None:
        plot_kwargs = {"cmap": "viridis", "interpolation": "nearest"}
    if cbar_kwargs is None:
        cbar_kwargs = {"shrink": 0.8, "aspect": 20}

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            da = list_ds[i][variable]
            img = _get_patch_image(da.data)
            ax.imshow(img, **plot_kwargs)
            _remove_axis(ax)
    return fig


def plot_som_feature_statistics(df_stats, feature):

    # Save df_summary
    grid_size = 10

    # Create a 2D array for x, y, and color values
    x_values = df_stats["col"].values
    y_values = df_stats["row"].values
    color_values = df_stats[feature].values

    # Create a grid for plotting
    grid = []

    # Fill the grid with color values at the corresponding positions
    for x, y, color in zip(x_values, y_values, color_values, strict=False):
        grid.append([x, y, color])

    # Separate x, y, and color values
    grid = np.array(grid)
    x = grid[:, 0]
    y = grid[:, 1]
    color = grid[:, 2]

    # Plot the heatmap
    plt.gca().invert_yaxis()
    plt.pcolor(
        x.reshape(grid_size, grid_size),
        y.reshape(grid_size, grid_size),
        color.reshape(grid_size, grid_size),
        cmap="viridis",
    )
    plt.colorbar(label=f"{feature} Mean")
    plt.xlabel("First ID")
    plt.ylabel("Second ID")
    plt.title("Variable Mean Heatmap")
    plt.show()
