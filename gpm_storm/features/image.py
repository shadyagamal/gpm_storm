#!/usr/bin/env python3

import cv2
import numpy as np
from skimage.measure import regionprops


def _calculate_mean_std_max_stats(data):
    data = data[~np.isnan(data)]

    if data.size == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    std = np.std(data)
    max_value = np.max(data)

    return mean, std, max_value


def _count_percentage_within_range(arr, vmin, vmax):
    # Mask pixels within the intensity range
    masked_pixels = np.logical_and(arr > vmin,  arr <= vmax)

    # Count the number of pixels within the intensity range
    pixels_in_range = np.sum(masked_pixels)

    # Calculate the percentage
    percentage = (pixels_in_range / np.nansum(arr > 0)) * 100
    return percentage


def _get_ellipse_major_minor_axis(label_arr):
    regions = regionprops(label_arr)
    if len(regions) == 0: 
        return np.nan, np.nan, 0
    largest_patch_index = np.argmax([region.area for region in regions])
    
    # Extract coordinates of pixels in the largest patch
    largest_patch_coords = regions[largest_patch_index].coords
    
    # Count number of pixels largest patch 
    n_pixels = largest_patch_coords.shape[0] 
    if n_pixels > 5:
        # Fit an ellipse to the coordinates
        ellipse = cv2.fitEllipse(np.array(largest_patch_coords))
        # Calculate the ratio between major and minor axes
        major_axis, minor_axis = ellipse[1]
    else: 
        major_axis = np.nan 
        minor_axis = np.nan
    return major_axis, minor_axis, n_pixels


def calculate_image_statistics(ds, patch_isel_dict): 
    """Define a function to calculate statistics for a patch."""
    
    ds_patch = ds.isel(patch_isel_dict)
    
    # Create a dictionary to store the dict_results
    dict_results = {}

    # Compute precipitation statistics
    dict_results["precipitation_average"] = np.nanmean(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_std"] = np.nanstd(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_pixel"] = np.nansum(ds_patch["precipRateNearSurface"].data > 0)
    dict_results["center_precipitation_pixel"] = np.nansum(ds_patch["precipRateNearSurface"][10:-10, 10:-10].data > 0)
    dict_results["precipitation_sum"] = np.nansum(ds_patch["precipRateNearSurface"].data)
    dict_results["precipitation_max"] = np.nanmax(ds_patch["precipRateNearSurface"].data)

    # Threshold values to iterate over, including 0 for count_over_0
    thresholds = [0, 1, 2, 5, 10, 20, 50, 80, 120]

    # Iterate over thresholds and count_over variables
    for threshold in thresholds: 
        arr = ds_patch["precipRateNearSurface"].data
        # Count rainy areas for a given threshold
        # - 0 is counted as a patches 
        count_patches, labeled_image = cv2.connectedComponents(
            (arr > threshold).astype(np.uint8), 
            connectivity=8
        )
        dict_results[f"count_rainy_areas_over_{threshold}"] = count_patches -1
       
        # Count rainy pixels for the current threshold
        count_pixels = np.sum(arr > threshold)
        mean_intensity = np.mean(arr[arr > threshold])
        dict_results[f"mean_for_rainy_pixels_over_{threshold}"] = mean_intensity
        dict_results[f"count_rainy_pixels_over_{threshold}"] = count_pixels
                
        # Compute aspect ratio 
        if threshold in [0, 1, 2, 5, 10, 20]:
            major_axis, minor_axis, n_patch_pixels = _get_ellipse_major_minor_axis(labeled_image)
            aspect_ratio = major_axis / minor_axis
            dict_results[f"major_axis_largest_patch_over_{threshold}"] = major_axis
            dict_results[f"minor_axis_largest_patch_over_{threshold}"] = minor_axis
            dict_results[f"aspect_ratio_largest_patch_over_{threshold}"] = aspect_ratio
            dict_results[f"count_rainy_pixels_in_patch_over_{threshold}"] = n_patch_pixels
 
    # Count percentage of values within a given value range
    intensity_ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20)]
    for vmin, vmax in intensity_ranges:
        percentage = _count_percentage_within_range(ds_patch["precipRateNearSurface"].data, 
                                                    vmin=vmin, vmax=vmax)
        dict_results[f"percentage_rainy_pixels_between_{vmin}_and_{vmax}"] = percentage

    # Compute reflectivity-related statistics
    ds_patch["REFCH"] = ds_patch.gpm_api.retrieve("REFCH").compute()
    ds_patch["echodepth18"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=18, mask_liquid_phase = True).compute()
    ds_patch["echodepth30"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=30, mask_liquid_phase = True).compute()
    ds_patch["echodepth50"] = ds_patch.gpm_api.retrieve("EchoDepth", threshold=50, mask_liquid_phase = True).compute()
    ds_patch["echotopheight18"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=18).compute()
    ds_patch["echotopheight30"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=30).compute()
    ds_patch["echotopheight50"] = ds_patch.gpm_api.retrieve("EchoTopHeight", threshold=50).compute()

    # Calculate mean/std/statistics for each variable with NaN values excluded and store in the dictionary
    variables = [
        "REFCH",
        "echodepth18",
        "echodepth30",
        "echodepth50",
        "echotopheight30",
        "echotopheight50",
    ]
    for variable in variables:
        mean, std, max_value = _calculate_mean_std_max_stats(ds_patch[variable].data)
        dict_results[f"{variable}_mean"] = mean
        dict_results[f"{variable}_std"] = std
        dict_results[f"{variable}_max"] = max_value

    # Add image patch identifiers (image center)
    along_track_slice = patch_isel_dict['along_track']
    dict_results["along_track_start"] = along_track_slice.start
    dict_results["along_track_end"] = along_track_slice.stop
    dict_results["gpm_granule_id"] = int(ds_patch["gpm_granule_id"][0].data)
    dict_results["time"] = ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)].data
    dict_results["sunLocalTime"] = float(ds_patch["sunLocalTime"][round(ds_patch["sunLocalTime"].data.shape[0] / 2)][round(ds_patch["sunLocalTime"].data.shape[1] / 2)].data)
    dict_results["lon"] = float(ds_patch["lon"][round(ds_patch["lon"].data.shape[0] / 2)][round(ds_patch["lon"].data.shape[1] / 2)].data)
    dict_results["lat"] = float(ds_patch["lat"][round(ds_patch["lat"].data.shape[0] / 2)][round(ds_patch["lat"].data.shape[1] / 2)].data)
    
    if along_track_slice.start == 0:
        dict_results["flag_granule_change"] = 1
    else: 
        dict_results["flag_granule_change"] = 0

    return dict_results
