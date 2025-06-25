#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:15:19 2025

@author: gamal
"""
def get_som_colormap(varname: str) -> str:
    """
    Returns the appropriate colormap for a given SOM variable name.
    """

    # Precipitation-related variables
    if varname.startswith("P_") or varname.startswith("MP_"):
        return "rain_r"

    # Morphology variables
    elif varname.startswith(("MA_", "MiA_", "AR_")):
        return "dense_r"

    # Reflectivity variables
    elif varname.startswith(("REFC_", "REFCH_", "ED", "ETH", "LCC_", "ICC_", "CC_")):
        return "eclipse"

    # Temperature (if present — placeholder, extend as needed)
    elif varname.lower().startswith("temp") or varname.lower().endswith("_temp"):
        return "sunset"

    # Land-type or catch-all (can refine as you grow the dataset)
    else:
        return "dense_r"
    
import pycolorbar
from pycolorbar import get_plot_kwargs  # noqa
colormaps = pycolorbar.colormaps
colorbars = pycolorbar.colorbars
from mpl_toolkits.axes_grid1 import make_axes_locatable


for col in num_df.columns:
    variable=col
    color = get_som_colormap(variable)
    cmap = colormaps.get_cmap(color)
    mean_values = np.full((10, 10), np.nan)
    
    for row in range(10):
        for col in range(10):
            df_node = arr_df[row, col]
            if not df_node.empty:
                mean_val = df_node[variable].mean()
                mean_values[row, col] = mean_val
        
    
    
    
    masked_array = np.ma.masked_invalid(mean_values)
    fig, ax = plt.subplots(figsize=(8, 8))
    p = ax.imshow(masked_array, cmap=cmap, origin="upper")
    
    # Add colorbar with larger label font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.2)
    cbar = plt.colorbar(p, cax=cax)
    cbar.set_label(f"Mean {variable}", fontsize=16)  
    cbar.ax.tick_params(labelsize=16)  
    
    # Labels, title, and ticks with larger fonts
    ax.set_title(f"Mean {variable} per SOM Node", fontsize=18)
    ax.set_xlabel("SOM Column", fontsize=16)
    ax.set_ylabel("SOM Row", fontsize=16)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.tick_params(axis='both', labelsize=16) 
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()


