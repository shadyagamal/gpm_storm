#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:00:20 2025

@author: gamal
"""

import gpm # type: ignore
import numpy as np 
import ximage # noqa
from gpm.io.local import get_local_filepaths
from scipy.ndimage import label as ndi_label


base_dir = None 
base_dir = "/ltenas2/data/GPM"

filepaths = get_local_filepaths(base_dir=base_dir, product="2A-DPR", version=7, product_type="RS")
filepath = filepaths[2]


variables = [
    "sunLocalTime",
    "airTemperature",
    "precipRate",
    "zFactorFinal",
    "zFactorMeasured",
    "precipRateNearSurface",
    "precipRateESurface",
    "precipRateESurface2",
    "zFactorFinalESurface",
    "zFactorFinalNearSurface",
    "heightZeroDeg",
    "binEchoBottom",
    "landSurfaceType",
]

def compute_CC_stats(ds_patch, threshold):
    ds_patch["REFC"] = ds_patch.gpm.retrieve("REFC")
    ds_patch["REFC"].gpm.plot_image()
    # Initialisation des statistiques
    stats = {
        f"CC_{threshold}_count": 0,
        f"LCC_{threshold}_mean": np.nan,
        f"LCC_{threshold}_max": np.nan,
        f"LCC_{threshold}_std": np.nan,
        f"ICC_{threshold}_mean": np.nan,
        f"ICC_{threshold}_max": np.nan,
        f"ICC_{threshold}_std": np.nan,
    }
    try:
        # Étiquetage des CC en fonction de l'intensité maximale
        xr_obj_intense = ds_patch["REFC"].ximage.label(
            min_value_threshold=threshold,
            min_area_threshold=2,
            footprint=0,
            sort_by="maximum",
            sort_decreasing=True,
            label_name="label",
        )
        xr_obj_intense["label"].gpm.plot_image()
        # Étiquetage des CC en fonction de la taille
        xr_obj_large = ds_patch["REFC"].ximage.label(
            min_value_threshold=threshold,
            min_area_threshold=2,
            footprint=0,
            sort_by="area",
            sort_decreasing=True,
            label_name="label",
        )
        stats[f"CC_{threshold}_count"] += xr_obj_large["label"].max().item()
        xr_obj_large["label"].gpm.plot_image()
        
        if (xr_obj_intense["label"] == 1).any():
            cc_intense_mask = xr_obj_intense["label"] == 1
            cc_intense_values = ds_patch["REFC"].where(cc_intense_mask).values.flatten()
            cc_intense_values = cc_intense_values[~np.isnan(cc_intense_values)]  
    
            if cc_intense_values.size > 0:
                stats[f"ICC_{threshold}_mean"] = np.nanmean(cc_intense_values)
                stats[f"ICC_{threshold}_max"] = np.nanmax(cc_intense_values)
                stats[f"ICC_{threshold}_std"] = np.nanstd(cc_intense_values)
    
        if (xr_obj_large["label"] == 1).any():
            cc_large_mask = xr_obj_large["label"] == 1
            cc_large_values = ds_patch["REFC"].where(cc_large_mask).values.flatten()
            cc_large_values = cc_large_values[~np.isnan(cc_large_values)]  
    
            if cc_large_values.size > 0:
                stats[f"LCC_{threshold}_mean"] = np.nanmean(cc_large_values)
                stats[f"LCC_{threshold}_max"] = np.nanmax(cc_large_values)
                stats[f"LCC_{threshold}_std"] = np.nanstd(cc_large_values)
    except Exception:
        return stats
    return stats 



ds = gpm.open_granule_dataset(filepath, 
                              scan_mode="FS",
                              chunks={})

da = ds["precipRateNearSurface"].compute()
xr_obj = da.ximage.label(
    min_value_threshold=0.05,
    max_value_threshold=np.inf,
    min_area_threshold=5,
    max_area_threshold=np.inf,
    footprint=5,
    sort_by="area",
    sort_decreasing=True,
    label_name="label",
)

# xr_obj["label"].ximage.plot_labels()

label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
    label_name="label",
    patch_size=(49, 49),
    variable="precipRateNearSurface",
    n_patches=2,
    n_labels=None,
    labels_id=None,
    padding=0,
    centered_on="max",
    partitioning_method=None,
    debug=False,
)



for isel_dict in label_isel_dict.values():
    ds_patch = ds.isel(**isel_dict[0]).compute()
    

# ds_patch["REFC"] = ds_patch.gpm.retrieve("REFC")
# ds_patch["REFC"].gpm.plot_image()
# ds_patch["zFactorFinalNearSurface"].sel(radar_frequency="Ku").gpm.plot_image()

# ds_patch["zFactorMeasured"].sel(radar_frequency="Ku").isel(cross_track=17).gpm.plot_cross_section()

# da =  ds_patch["zFactorFinal"].sel(radar_frequency="Ku")
# da = ds_patch["zFactorMeasured"].sel(radar_frequency="Ku")
# da = da.where(da > 14)
    
# for i in range(0, 49):
#     # p = ds_patch["zFactorFinal"].sel(radar_frequency="Ku").isel(cross_track=i).gpm.plot_cross_section()

#     p = da.isel(cross_track=i).gpm.plot_cross_section()
#     p.axes.set_title(i)
#     plt.show()
    
# for i in range(0, 49):
#     # p = ds_patch["zFactorFinal"].sel(radar_frequency="Ku").isel(cross_track=i).gpm.plot_cross_section()

#     p = da.isel(along_track=i).gpm.plot_cross_section()
#     p.axes.set_title(i)
#     plt.show()
    
# xr_obj = ds_patch["REFC"].ximage.label(
#         min_value_threshold=30,
#         max_value_threshold=np.inf,
#         min_area_threshold=2,
#         max_area_threshold=np.inf,
#         footprint=0,
#         sort_by="maximum",
#         sort_decreasing=True,
#         label_name="label",
#     )
# xr_obj["label"].gpm.plot_image()
# (xr_obj["label"] ==1).gpm.plot_image()





