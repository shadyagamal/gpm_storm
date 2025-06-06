#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:47:35 2025

@author: gamal
"""
from scipy.ndimage import label
def compute_wcc_flags_3d(z_volume, threshold, area_thresh_km2, pixel_area_km2=25):
    """
    Detect WCC by checking if any 3D convective object (above threshold)
    has a horizontal *contiguous* 2D slice with sufficient area.
    """
    binary_mask = z_volume > threshold
    labeled_3d, num_features = label(binary_mask)  # 3D connected objects

    for i in range(1, num_features + 1):
        obj_mask = (labeled_3d == i)

        for z in range(obj_mask.shape[0]):  # for each altitude
            horizontal_slice = obj_mask[z]

            # Label 2D connected regions in the horizontal slice
            labeled_2d, num_2d = label(horizontal_slice)

            for j in range(1, num_2d + 1):
                region = (labeled_2d == j)
                area_km2 = region.sum() * pixel_area_km2
                if area_km2 >= area_thresh_km2:
                    return True  # WCC found
    return False


# --- Main script ---

wcc_flag = []
node_33 = df_bmu[(df_bmu["row"] == 5) & (df_bmu["col"] == 6)]
sample = node_33.iloc[1]  # get your sample row

# Load DPR data with gpm-api
ds = gpm.open_dataset(
    product="2A-DPR",
    product_type="RS",
    version=7,
    start_time=sample["time"] - datetime.timedelta(minutes=1),
    end_time=sample["time"] + datetime.timedelta(minutes=1),
    scan_mode="FS",
    chunks={},
    variables=["zFactorFinal", "height", "typePrecip", "heightZeroDeg"]
)


ds_patch = ds.gpm.sel(gpm_id=slice(sample["gpm_id_start"], sample["gpm_id_end"])).compute()
ds_patch_footprints = ds_patch.stack(footprints=["cross_track", "along_track"])
z_ku = ds_patch_footprints["zFactorFinal"].sel(radar_frequency="Ku")
z_cube = z_ku.unstack("footprints").transpose("range", "cross_track", "along_track").values

# Compute WCC flags
wcc_moderate = compute_wcc_flags_3d(z_cube, threshold=30, area_thresh_km2=800)
wcc_strong = compute_wcc_flags_3d(z_cube, threshold=40, area_thresh_km2=1000)


# binary_mask = z_cube > 30
# labeled, num_features = label(binary_mask)  

# for i in range(1, num_features + 1):
#     obj = (labeled == i)

#     for z in range(obj.shape[0]):
#         horizontal_slice = obj[z]
#         area_km2 = horizontal_slice.sum() * pixel_area_km2
#         if area_km2 >= area_thresh_km2:

# # Save result
# wcc_flag.append({
#     "index": sample.name,
#     "wcc_moderate": wcc_moderate,
#     "wcc_strong": wcc_strong
# })