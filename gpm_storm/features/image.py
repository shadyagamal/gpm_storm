import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import label as ndi_label
import gpm # type: ignore
import ximage # noqa
from gpm.io.local import get_local_filepaths



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
    "flagPrecip", 
    "typePrecip"
]

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
    CCthresholds = [30,40]
    for threshold in CCthresholds:
        CC_stats = compute_CC_stats(ds_patch, threshold)
        stats.update(CC_stats)
        
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

def compute_CC_stats(ds_patch, threshold):
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

def calculate_image_statistics2(ds_patch):  
    """
    Compute statistics for an extracted precipitation patch.
    """
    ds_patch = ds_patch.compute()  
    precip_data = ds_patch["precipRateNearSurface"].data  

    # Initialize results dictionary 
    # P = precipitation pixel
    stats = {
        "P_mean": np.nanmean(precip_data),
        "P_std": np.nanstd(precip_data),
        "P_sum": np.nansum(precip_data > 0),
        "P_center_sum": np.nansum(precip_data[10:-10, 10:-10] > 0),
        "P_max": np.nanmax(precip_data),
    }

    thresholds = [0, 1, 2, 5, 10, 20, 50, 80, 120]  
    for threshold in thresholds:
        labeled_image, count_patches = ndi_label(precip_data > threshold)
        stats[f"PP_count_over_{threshold}"] = count_patches
        stats[f"P_sum_over_{threshold}"] = np.sum(precip_data > threshold)

        if np.any(precip_data > threshold):
            stats[f"P_mean_over_{threshold}"] = np.mean(precip_data[precip_data > threshold])

        # Compute aspect ratio for lower thresholds 
        # MA = major axis
        # MiA = minor axis
        # LP = largest patch
        # AR = aspect ratio
        if threshold in [0, 1, 2, 5, 10, 20]:
            major_axis, minor_axis, n_patch_pixels = _get_ellipse_major_minor_axis(labeled_image)
            aspect_ratio = major_axis / minor_axis if minor_axis else np.nan
            stats.update({
                f"MA_LP_over_{threshold}": major_axis,
                f"MiA_LP_over_{threshold}": minor_axis,
                f"AR_LP_over_{threshold}": aspect_ratio,
                f"P_count_over_{threshold}": n_patch_pixels,
            })


    intensity_ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20)]
    for vmin, vmax in intensity_ranges:
        stats[f"P_%_between_{vmin}_{vmax}"] = _count_percentage_within_range(precip_data, vmin, vmax)

    # Additional meteorological variables  
    variables = ["REFCH", "echodepth18", "echodepth30", "echodepth50", "echotopheight20", "echotopheight30", "echotopheight40", "echotopheight50"]
    for var in variables:
        if var in ds_patch:
            mean, std, max_val = _calculate_mean_std_max_stats(ds_patch[var].data)
            stats.update({
                f"{var}_mean": mean,
                f"{var}_std": std,
                f"{var}_max": max_val,
            })

    if "heightZeroDeg" in ds_patch:
        warm_rain_mask = (precip_data.sum(axis=0) > 0) & (ds_patch["heightZeroDeg"].data > 0)
        stats["WR_%"] = 100 * np.sum(warm_rain_mask) / precip_data.size
    # V = Virga 
    if "flagPrecip" in ds_patch:
        virga_mask = (ds_patch["precipRateNearSurface"] == 0).data & (ds_patch["flagPrecip"] > 0).data
        stats["V_%"] = 100 * np.sum(virga_mask) / precip_data.size
        
    # PT = precipitation type
    if "precip_types" in ds_patch:
        precip_types = ds_patch["precip_types"].data
        stats.update({
                "PT_strat_%": 100 * np.sum(precip_types == 1) / precip_types.size,
                "PT_conve_%": 100 * np.sum(precip_types == 2) / precip_types.size,
                "PT_other_%": 100 * np.sum(precip_types == 3) / precip_types.size,
            })

    #LST = land surface type
    if "landSurfaceType" in ds_patch:
        land_surface = ds_patch["landSurfaceType"].data
        stats.update({
                "LST_Ocean_%": 100 * np.sum(land_surface == 0) / precip_data.size,
                "LST_Land_%": 100 * np.sum(land_surface == 1) / precip_data.size,
                "LST_Coast_%": 100 * np.sum(land_surface == 2) / precip_data.size,
                "LST_InlandW_%": 100 * np.sum(land_surface == 3) / precip_data.size,
            })
        
    CCthresholds = [30,40]
    for threshold in CCthresholds:
        CC_stats = compute_CC_stats(ds_patch, threshold)
        stats.update(CC_stats)
        
    # Metadata for tracking  
    stats.update({
        "gpm_granule_id": int(ds_patch["gpm_granule_id"][0].data),
        "time": (ds_patch["time"][round(ds_patch["time"].data.shape[0] / 2)]).data,
        "sunLocalTime": float(ds_patch["sunLocalTime"].isel(along_track=ds_patch.sizes["along_track"] // 2, cross_track=ds_patch.sizes["cross_track"] // 2).values),
        "lon": float(ds_patch["lon"].isel(along_track=ds_patch.sizes["along_track"] // 2, cross_track=ds_patch.sizes["cross_track"] // 2).values),
        "lat": float(ds_patch["lat"].isel(along_track=ds_patch.sizes["along_track"] // 2, cross_track=ds_patch.sizes["cross_track"] // 2).values),
        "flag_granule_change": 1 if ds_patch.along_track.values[0] == 0 else 0,
    })
    
    

    return stats


ds = gpm.open_granule_dataset(filepath, 
                              scan_mode="FS",
                              variables=variables,
                              chunks={})


ds["precip_types"] = ds.gpm.retrieve("flagPrecipitationType", method="major_rain_type")
ds["REFC"] = ds.gpm.retrieve("REFC")
ds["REFCH"] = ds.gpm.retrieve("REFCH")


thresholds = [18, 30, 50]  
for threshold in thresholds:
    ds[f"echodepth{threshold}"] = ds.gpm.retrieve("EchoDepth", threshold=threshold, mask_liquid_phase=True)

thresholds = [20, 30, 40, 50]
for threshold in thresholds:
    ds[f"echotopheight{threshold}"] = ds.gpm.retrieve("EchoTopHeight", threshold=threshold)



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
    stats = calculate_image_statistics2(ds_patch)
