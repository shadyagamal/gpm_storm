import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import label as ndi_label


def _calculate_mean_std_max_stats(data):
    data = data[~np.isnan(data)]
    if data.size == 0:
        return np.nan, np.nan, np.nan
    return np.mean(data), np.std(data), np.max(data)


def _count_percentage_within_range(arr, vmin, vmax):
    masked_pixels = np.logical_and(arr > vmin, arr <= vmax)
    pixels_in_range = np.sum(masked_pixels)
    return (pixels_in_range / np.nansum(arr > 0)) * 100


def _get_ellipse_major_minor_axis(label_arr):
    regions = regionprops(label_arr)
    if len(regions) == 0: 
        return np.nan, np.nan, 0

    largest_patch = max(regions, key=lambda r: r.area)
    n_pixels = largest_patch.area
    
    if n_pixels > 5:
        major_axis = largest_patch.major_axis_length
        minor_axis = largest_patch.minor_axis_length
    else: 
        major_axis, minor_axis = np.nan, np.nan

    return major_axis, minor_axis, n_pixels


def calculate_image_statistics(ds_patch): 
    ds_patch = ds_patch.compute()  
    precip_data = ds_patch["precipRateNearSurface"].data

    # Initialize results dictionary
    stats = {
        "precipitation_average": np.nanmean(precip_data),
        "precipitation_std": np.nanstd(precip_data),
        "precipitation_pixel": np.nansum(precip_data > 0),
        "center_precipitation_pixel": np.nansum(precip_data[10:-10, 10:-10] > 0),
        "precipitation_sum": np.nansum(precip_data),
        "precipitation_max": np.nanmax(precip_data),
    }

    # Rainy area and intensity analysis
    thresholds = [0, 1, 2, 5, 10, 20, 50, 80, 120]
    for threshold in thresholds:
        labeled_image, count_patches = ndi_label(precip_data > threshold)
        stats[f"count_rainy_areas_over_{threshold}"] = count_patches
        stats[f"count_rainy_pixels_over_{threshold}"] = np.sum(precip_data > threshold)

        if np.any(precip_data > threshold):
            stats[f"mean_for_rainy_pixels_over_{threshold}"] = np.mean(precip_data[precip_data > threshold])

        # Compute aspect ratio for lower thresholds
        if threshold in [0, 1, 2, 5, 10, 20]:
            major_axis, minor_axis, n_patch_pixels = _get_ellipse_major_minor_axis(labeled_image)
            aspect_ratio = major_axis / minor_axis if minor_axis else np.nan
            stats.update({
                f"major_axis_largest_patch_over_{threshold}": major_axis,
                f"minor_axis_largest_patch_over_{threshold}": minor_axis,
                f"aspect_ratio_largest_patch_over_{threshold}": aspect_ratio,
                f"count_rainy_pixels_in_patch_over_{threshold}": n_patch_pixels,
            })

    # Percentage of rainy pixels in intensity ranges
    intensity_ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20)]
    for vmin, vmax in intensity_ranges:
        stats[f"percentage_rainy_pixels_between_{vmin}_and_{vmax}"] = _count_percentage_within_range(precip_data, vmin, vmax)

    # Additional meteorological variables
    variables = ["REFCH", "echodepth30", "echodepth50", "echotopheight30", "echotopheight50"]
    for var in variables:
        if var in ds_patch:
            mean, std, max_val = _calculate_mean_std_max_stats(ds_patch[var].data)
            stats.update({
                f"{var}_mean": mean,
                f"{var}_std": std,
                f"{var}_max": max_val,
            })

    # Metadata for tracking
    stats.update({
        "gpm_granule_id": int(ds_patch["gpm_granule_id"][0].data),
        "time": ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)].data,
        "sunLocalTime": float(ds_patch["sunLocalTime"][round(ds_patch["sunLocalTime"].data.shape[0] / 2)][round(ds_patch["sunLocalTime"].data.shape[1] / 2)].data),
        "lon": float(ds_patch["lon"].isel(along_track=ds_patch.sizes["along_track"] // 2, cross_track=ds_patch.sizes["cross_track"] // 2).values),
        "lat": float(ds_patch["lat"].isel(along_track=ds_patch.sizes["along_track"] // 2, cross_track=ds_patch.sizes["cross_track"] // 2).values),
        "flag_granule_change": 1 if ds_patch.along_track.values[0] == 0 else 0,
    })

    return stats
