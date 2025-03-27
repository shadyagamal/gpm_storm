import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import label as ndi_label
import gpm # type: ignore
import ximage # noqa
from gpm.io.local import get_local_filepaths



# base_dir = None 
# base_dir = "/ltenas2/data/GPM"

# filepaths = get_local_filepaths(base_dir=base_dir, product="2A-DPR", version=7, product_type="RS")
# filepath = filepaths[2]

# variables = [
#     "sunLocalTime",
#     "airTemperature",
#     "precipRate",
#     "zFactorFinal",
#     "zFactorMeasured",
#     "precipRateNearSurface",
#     "precipRateESurface",
#     "precipRateESurface2",
#     "zFactorFinalESurface",
#     "zFactorFinalNearSurface",
#     "heightZeroDeg",
#     "binEchoBottom",
#     "landSurfaceType",
#     "flagPrecip", 
#     "typePrecip"
# ]

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


def calculate_image_statistics(ds_patch):  
    """
    Compute statistics for an extracted precipitation patch.
    """
    ds_patch = ds_patch.compute()  # compute only variables needed
    
    precip_data = ds_patch["precipRateNearSurface"].data  

    # Initialize results dictionary 
    # P = precipitation pixel
    stats = {
        "P_mean": np.nanmean(precip_data), # [mm/h]
        "P_std": np.nanstd(precip_data), # [mm/h]
        "P_center_count": np.nansum(precip_data[10:-10, 10:-10] > 0), # [#pixels]
        "P_sum": np.nansum(precip_data), # [mm/h]
        "P_max": np.nanmax(precip_data), # [mm/h]
        "P_count" : np.nansum(precip_data>0), # [#pixels]
    }

    # Compute statistics for different precipitation thresholds
    thresholds = [0, 1, 2, 5, 10, 20, 50, 80, 120]  
    precip_masks = {t: precip_data > t for t in thresholds}
    
    # PP = precipitation patch
    for threshold, mask in precip_masks.items():
        labeled_image, count_patches = ndi_label(mask)
        stats.update({
            f"P_GT{threshold}_regions": 0, # [#cores]
            f"P_GT{threshold}_count": 0, # [#pixels]
            f"P_GT{threshold}_mean": np.nan, # [mm/hr]
            f"P_GT{threshold}_sum": np.nan, # [mm/hr]
            f"P_GT{threshold}_min": np.nan, # [mm/hr]
        })
        if any(precip_data[mask]):
            stats.update({
                f"P_GT{threshold}_regions": count_patches, # [#cores]
                f"P_GT{threshold}_count": np.sum(mask), # [#pixels]
                f"P_GT{threshold}_mean": np.nanmean(precip_data[mask]), # [mm/hr]
                f"P_GT{threshold}_sum": np.nansum(precip_data[mask]), # [mm/hr]
                f"P_GT{threshold}_min": np.nanmin(precip_data[mask]), # [mm/hr]
            })

        # Compute aspect ratio for lower thresholds 
        # MA = major axis
        # MiA = minor axis
        # LP = largest patch
        # AR = aspect ratio
        if threshold <= 20:  # Aspect ratio calculations for lower thresholds
            major_axis, minor_axis, n_patch_pixels = _get_ellipse_major_minor_axis(labeled_image)
            aspect_ratio = major_axis / minor_axis if minor_axis else np.nan
            stats.update({
                f"MA_LP_GT_{threshold}": major_axis, # [#pixels]
                f"MiA_LP_GT_{threshold}": minor_axis, # [#pixels]
                f"AR_LP_GT_{threshold}": aspect_ratio, # [ratio]
            })

    # Compute percentage of precipitation in intensity ranges
    intensity_ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 300)]
    stats.update({
        f"P_%_between_{vmin}_{vmax}": _count_percentage_within_range(precip_data, vmin, vmax)
        for vmin, vmax in intensity_ranges
    })
    
    # Data retrieval
    ds_patch["precip_types"] = ds_patch.gpm.retrieve("flagPrecipitationType", method="major_rain_type")
    ds_patch["REFC"] = ds_patch.gpm.retrieve("REFC")
    ds_patch["REFCH"] = ds_patch.gpm.retrieve("REFCH")

    thresholds = [30, 40, 50]  
    for threshold in thresholds:
        ds_patch[f"ED{threshold}_SOLID"] = ds_patch.gpm.retrieve("EchoDepth", threshold=threshold, mask_liquid_phase=True)
        ds_patch[f"ED{threshold}_FULL"] = ds_patch.gpm.retrieve("EchoDepth", threshold=threshold, mask_liquid_phase=False)
        ds_patch[f"ETH{threshold}"] = ds_patch.gpm.retrieve("EchoTopHeight", threshold=threshold)
    
    # Compute statistics for additional meteorological variables # REFCH [dbz] else [m]
    variables = ["REFC", "REFCH",
                 "ED30_SOLID", "ED30_FULL","ETH30",
                 "ED40_SOLID", "ED40_FULL","ETH40", 
                 "ED50_SOLID", "ED50_FULL","ETH50", 
    ] 
    stats.update({
        f"{var}_{stat}": val
        for var in variables if var in ds_patch
        for stat, val in zip(["mean", "std", "max"], _calculate_mean_std_max_stats(ds_patch[var].data))
    })

    # Compute Warm Rain (WR) fraction
    if "heightZeroDeg" in ds_patch:
        warm_rain_mask = (precip_data.sum(axis=0) > 0) & (ds_patch["heightZeroDeg"].data > 0)
        stats["WR_frac"] = np.nanmean(warm_rain_mask)

    # Compute Virga (V) fraction
    if "flagPrecip" in ds_patch:
        virga_mask = (ds_patch["precipRateNearSurface"].data == 0) & (ds_patch["flagPrecip"].data > 0)
        stats["V_frac"] = np.nanmean(virga_mask)

    # Compute Precipitation Type (PT) fraction
    if "precip_types" in ds_patch:
        precip_types = ds_patch["precip_types"].data
        stats.update({
            "PT_strat_frac": np.nanmean(precip_types == 1),
            "PT_conve_frac": np.nanmean(precip_types == 2),
            "PT_other_frac": np.nanmean(precip_types == 3),
        })

    # Compute Land Surface Type (LST) fraction
    if "landSurfaceType" in ds_patch:
        land_surface = ds_patch["landSurfaceType"].data
        stats.update({
            "Ocean_fraction": np.nanmean(land_surface == 0),
            "Land_fraction": np.nanmean(land_surface == 1),
            "Coast_fraction": np.nanmean(land_surface == 2),
            "InlandW_fraction": np.nanmean(land_surface == 3),
        })

    # Compute Cloud Classification (CC) statistics
    for threshold in [30, 40]:
        stats.update(compute_CC_stats(ds_patch, threshold))
    
     
    # Metadata for tracking  
    center_idx = {
        "along_track": ds_patch.sizes["along_track"] // 2,
        "cross_track": ds_patch.sizes["cross_track"] // 2,
    }
    stats.update({
        "Air_temp": np.nanmean(ds_patch["airTemperature"].gpm.slice_range_at_bin(ds_patch["binEchoBottom"])),
        "gpm_granule_id": int(ds_patch["gpm_granule_id"][0].data),
        "time": ds_patch["time"].isel(along_track=center_idx["along_track"]).data,
        "sunLocalTime": float(ds_patch["sunLocalTime"].isel(**center_idx).values),
        "lon": float(ds_patch["lon"].isel(**center_idx).values),
        "lat": float(ds_patch["lat"].isel(**center_idx).values),
        "flag_granule_change": int(ds_patch.along_track.values[0] == 0),
        "gpm_id_start": ds_patch["gpm_id"].isel(along_track=0).values.item(),
        "gpm_id_end":  ds_patch["gpm_id"].isel(along_track=-1).values.item(),
    })

    return stats

# ds = gpm.open_granule_dataset(filepath, 
#                               scan_mode="FS",
#                               variables=variables,
#                               chunks={})

# ds["precip_types"] = ds.gpm.retrieve("flagPrecipitationType", method="major_rain_type")
# ds["REFC"] = ds.gpm.retrieve("REFC")
# ds["REFCH"] = ds.gpm.retrieve("REFCH")


# thresholds = [18, 30, 50]  
# for threshold in thresholds:
#     ds[f"echodepth{threshold}"] = ds.gpm.retrieve("EchoDepth", threshold=threshold, mask_liquid_phase=True)

# thresholds = [20, 30, 40, 50]
# for threshold in thresholds:
#     ds[f"echotopheight{threshold}"] = ds.gpm.retrieve("EchoTopHeight", threshold=threshold)


# da = ds["precipRateNearSurface"].compute()
# xr_obj = da.ximage.label(
#     min_value_threshold=0.05,
#     max_value_threshold=np.inf,
#     min_area_threshold=5,
#     max_area_threshold=np.inf,
#     footprint=5,
#     sort_by="area",
#     sort_decreasing=True,
#     label_name="label",
# )

# label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
#     label_name="label",
#     patch_size=(49, 49),
#     variable="precipRateNearSurface",
#     n_patches=2,
#     n_labels=None,
#     labels_id=None,
#     padding=0,
#     centered_on="max",
#     partitioning_method=None,
#     debug=False,
# )

# for isel_dict in label_isel_dict.values():
#     ds_patch = ds.isel(**isel_dict[0]).compute()
    
# import time 
# start_time=time.time()
# stats3 = calculate_image_statistics(ds_patch)
# end_time=time.time()
# stats3
# elapsed_time = end_time - start_time
# print(f"Time taken for patch computation: {elapsed_time:.4f} seconds")

