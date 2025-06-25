# --- IMPORTS ---
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import glob
from gpm_storm.som.experiments import load_som
from gpm_storm.som.io import create_som_df_array
import gpm_geo

def find_zarr_file_for_patch(row, zarr_directory, filename_pattern="*.zarr"):
    granule_id, patch_id = str(row["gpm_granule_id"]), row["patch_id"]
    time = pd.to_datetime(row["time"])
    
    for offset in [0, -1]:
        t = time + pd.DateOffset(months=offset)
        path = os.path.join(zarr_directory, f"{t.year:04d}/{t.month:02d}", filename_pattern)
        for zarr_file in glob.glob(path):
            if granule_id in os.path.basename(zarr_file):
                ds = xr.open_zarr(zarr_file)
                if patch_id < ds.sizes["patch"]:
                    return zarr_file, ds.isel(patch=patch_id)
    print(f"No Zarr for granule_id {granule_id} at {time.strftime('%Y-%m')}")
    return None, None


def build_som_sample_array(df, som):
    arr_df = create_som_df_array(som=som, df=df)
    arr_ds_gpm_patch = np.empty(arr_df.shape, dtype=object)
    arr_ds_gpm_geo_image = np.empty(arr_df.shape, dtype=object)

    for row in range(arr_df.shape[0]):
        for col in range(arr_df.shape[1]):
            df_node = arr_df[row, col]
            print(f"node({row},{col})")
            if not len(df_node):
                continue

            found_valid_patch = False
            random_indices = np.random.permutation(len(df_node))

            for i in random_indices:
                print(i)
                sample = df_node.iloc[i]

                try:
                    _, patch_ds = find_zarr_file_for_patch(sample, zarr_dir)
                    ds_gpm_patch = patch_ds
                    ds_gpm_patch["gpm_granule_id"] = xr.ones_like(
                        ds_gpm_patch["gpm_along_track_id"]
                    ) * ds_gpm_patch["gpm_granule_id"]

                    ds_gpm_patch = ds_gpm_patch.compute()
                    ds_gpm_patch = ds_gpm_patch.sel(radar_frequency="Ku")

                    try:
                        gpm_geo_fpath = gpm_geo.find_files_matching_acquisitions(
                            satellite=satellite,
                            product="RAD",
                            timesteps=ds_gpm_patch.gpm.start_time
                        )[0]
                    except IndexError:
                        continue

                    ds_gpm_geo = gpm_geo.open_products(gpm_geo_fpath, chunks={})
                    try:
                        ds_gpm_geo_patch, ds_gpm_patch = gpm_geo.align_datasets(
                            ds_gpm_geo, ds_gpm_patch, cross_track=False)
                    except Exception:
                        continue  
                    try:
                        ds_gpm_geo_image = ds_gpm_geo_patch.gpm_geo.select_collocated_image()
                    except Exception:
                        continue 

                    arr_ds_gpm_patch[row, col] = ds_gpm_patch
                    arr_ds_gpm_geo_image[row, col] = ds_gpm_geo_image
                    found_valid_patch = True
                    print("YEAAAAAAAAAAAAAAH")
                    break

                except Exception as e:
                    print(e)
                    continue

            if not found_valid_patch:
                print(f"No valid GEO match found for node ({row}, {col})")

    return arr_df, arr_ds_gpm_patch, arr_ds_gpm_geo_image

import warnings
def build_som_sample_array_better(df, som):
    arr_df = create_som_df_array(som=som, df=df)
    arr_ds_gpm_patch = np.empty(arr_df.shape, dtype=object)
    arr_ds_gpm_geo_image = np.empty(arr_df.shape, dtype=object)

    for row in range(arr_df.shape[0]):
        for col in range(arr_df.shape[1]):
            df_node = arr_df[row, col]
            print(f"node({row},{col})")
            if not len(df_node):
                continue

            found_valid_patch = False
            random_indices = np.random.permutation(len(df_node))

            for i in random_indices:
                print(i)
                sample = df_node.iloc[i]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)  # Turn RuntimeWarnings into exceptions

                        # Try loading the GPM patch
                        _, patch_ds = find_zarr_file_for_patch(sample, zarr_dir)
                        ds_gpm_patch = patch_ds
                        ds_gpm_patch["gpm_granule_id"] = xr.ones_like(
                            ds_gpm_patch["gpm_along_track_id"]
                        ) * ds_gpm_patch["gpm_granule_id"]

                        ds_gpm_patch = ds_gpm_patch.compute()
                        ds_gpm_patch = ds_gpm_patch.sel(radar_frequency="Ku")

                        # Try to find and align GEO data
                        try:
                            gpm_geo_fpath = gpm_geo.find_files_matching_acquisitions(
                                satellite=satellite,
                                product="RAD",
                                timesteps=ds_gpm_patch.gpm.start_time
                            )[0]
                        except IndexError:
                            continue

                        ds_gpm_geo = gpm_geo.open_products(gpm_geo_fpath, chunks={})
                        try:
                            ds_gpm_geo_patch, ds_gpm_patch = gpm_geo.align_datasets(
                                ds_gpm_geo, ds_gpm_patch, cross_track=False)
                        except Exception:
                            continue  

                        try:
                            ds_gpm_geo_image = ds_gpm_geo_patch.gpm_geo.select_collocated_image()
                        except Exception:
                            continue 

                    # Store the result
                    arr_ds_gpm_patch[row, col] = ds_gpm_patch
                    arr_ds_gpm_geo_image[row, col] = ds_gpm_geo_image
                    found_valid_patch = True
                    print("YEAAAAAAAAAAAAAAH")
                    break

                except Exception as e:
                    print(f"Exception caught for patch {i}: {e}")
                    continue

            if not found_valid_patch:
                print(f"No valid GEO match found for node ({row}, {col})")

    return arr_df, arr_ds_gpm_patch, arr_ds_gpm_geo_image


som_name = "SOM_Pmean_>_1_with_random_init" 
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/0_Results") 
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}")
zarr_dir = "/ltenas2/data/GPM_STORM_DB/zarr"

satellite="GOES16"
product="RAD"
geo_composite = "cimss_true_color"
dummy_variable ="zFactorFinalNearSurface"
interpolation = "quadric"


df_bmu = pd.read_parquet(bmu_dir)
df_bmu["time"] = pd.to_datetime(df_bmu["time"])
df_bmu["year"] = df_bmu["time"].dt.year

df_bmu_2020 = df_bmu[
    (df_bmu["lon"] > -81) & 
    (df_bmu["lon"] < -35) & 
    (df_bmu["lat"] > -35) & 
    (df_bmu["lat"] < 12) & 
    (df_bmu["year"] == 2020)].copy()

som = load_som(som_dir=som_dir, som_name=som_name)
arr_df, arr_ds_gpm_patch, arr_ds_gpm_geo_image = build_som_sample_array(df_bmu_2020, som)
# plot_som_grid_samples(arr_ds, save_dir=figs_dir)


for row in range(arr_ds_gpm_geo_image.shape[0]):
    for col in range(arr_ds_gpm_geo_image.shape[1]):
        image = arr_ds_gpm_geo_image[row][col]
        image.gpm_geo.plot("true_color")
        

image = arr_ds_gpm_geo_image[9][8]
patch = arr_ds_gpm_patch[9][8]
image.gpm_geo.plot("true_color")
gpm_geo.plot_composite_and_gpm(
    ds_rad=image,
    ds_gpm=patch,
    gpm_variable="zFactorFinalNearSurface",
    geo_composite=geo_composite,
    interpolation=interpolation,
    plot_gpm=True,
    add_colorbar=True, 
    visible_colorbar=True,
)

# ds_gpm_geo_patch.gpm_geo.available_composites_names()
# ['24h_microphysics',
#  'airmass',
#  'ash',
#  'blowing_snow',
#  'cimss_cloud_type',
#  'cimss_cloud_type_raw',
#  'cimss_green',
#  'cimss_green_sunz',
#  'cimss_green_sunz_rayleigh',
#  'cimss_true_color',
#  'cimss_true_color_sunz',
#  'cimss_true_color_sunz_rayleigh',
#  'cira_day_convection',
#  'cira_fire_temperature',
#  'cloud_phase',
#  'cloud_phase_distinction',
#  'cloud_phase_distinction_raw',
#  'cloud_phase_raw',
#  'cloudtop',
#  'color_infrared',
#  'colorized_ir_clouds',
#  'convection',
#  'day_cloud_type',
#  'day_microphysics',
#  'day_microphysics_abi',
#  'day_microphysics_eum',
#  'dust',
#  'fire_temperature_awips',
#  'fog',
#  'geo_color',
#  'geo_color_background_with_low_clouds',
#  'geo_color_high_clouds',
#  'geo_color_low_clouds',
#  'geo_color_night',
#  'green',
#  'green_crefl',
#  'green_nocorr',
#  'green_raw',
#  'green_snow',
#  'highlight_C14',
#  'ir108_3d',
#  'ir_cloud_day',
#  'land_cloud',
#  'land_cloud_fire',
#  'natural_color',
#  'natural_color_nocorr',
#  'natural_color_raw',
#  'natural_color_raw_with_night_ir',
#  'night_fog',
#  'night_ir_alpha',
#  'night_ir_with_background',
#  'night_ir_with_background_hires',
#  'night_microphysics',
#  'night_microphysics_eum',
#  'overview',
#  'overview_raw',
#  'rocket_plume_day',
#  'rocket_plume_night',
#  'snow',
#  'snow_fog',
#  'so2',
#  'tropical_airmass',
#  'true_color',
#  'true_color_crefl',
#  'true_color_nocorr',
#  'true_color_raw',
#  'true_color_reproduction',
#  'true_color_reproduction_corr',
#  'true_color_reproduction_uncorr',
#  'true_color_with_night_fires',
#  'true_color_with_night_fires_nocorr',
#  'true_color_with_night_ir',
#  'true_color_with_night_ir_hires',
#  'water_vapors1',
#  'water_vapors2']


# sample = df_bmu_2020.iloc[756]
# _, patch_ds = find_zarr_file_for_patch(sample, zarr_dir)
# ds_gpm_patch = patch_ds
# ds_gpm_patch["gpm_granule_id"] = xr.ones_like(ds_gpm_patch["gpm_along_track_id"])*ds_gpm_patch["gpm_granule_id"] 

# ds_gpm_patch = ds_gpm_patch.compute()
# gpm_geo_fpath = gpm_geo.find_files_matching_acquisitions(satellite=satellite, 
#                                                          product="RAD", 
#                                                          timesteps=ds_gpm_patch.gpm.start_time)[0]
# ds_gpm_geo = gpm_geo.open_products(gpm_geo_fpath, chunks={})

# # Align GPM-GEO dataset to GPM patch
# ds_gpm_geo_patch, ds_gpm_patch = gpm_geo.align_datasets(ds_gpm_geo, ds_gpm_patch, cross_track=True)

# # Extract coincident GEO imagery
# ds_gpm_geo_image = ds_gpm_geo_patch.gpm_geo.select_collocated_image()

# # Put GPM data into memory 
# ds_gpm_patch = ds_gpm_patch.compute()
# ds_gpm_patch = ds_gpm_patch.sel(radar_frequency="Ku")
# ds_gpm_geo_image.gpm_geo.plot("cimss_true_color")

##----------------------------------------------------------.
#### - Display DPR
# - Surface reflectivity    
# gpm_geo.plot_composite_and_gpm(
#     ds_rad=ds_gpm_geo_image,
#     ds_gpm=ds_gpm_patch,
#     gpm_variable="zFactorFinalNearSurface",
#     geo_composite=geo_composite,
#     interpolation=interpolation,
#     plot_gpm=True,
#     add_colorbar=True, 
#     visible_colorbar=True,
# )

# gpm_geo.plot_composite_and_gpm(
#     ds_rad=ds_gpm_geo_image,
#     ds_gpm=ds_gpm_patch,
#     gpm_variable="zFactorFinalNearSurface",
#     geo_composite=geo_composite,
#     interpolation=interpolation,
#     plot_gpm=False,
#     add_colorbar=False, 
#     visible_colorbar=False,
# )


# ##----------------------------------------------------------.
# for i in range(ds_gpm_geo_patch.sizes["geo_acquisition"]):
#     ds_gpm_geo_patch.isel(geo_acquisition=i).gpm_geo.plot("true_color")
#     plt.show()
    
    





# def plot_geo_image_grid(arr_ds, var="gpm_geo_image", cmap=cmap, figsize=(15, 15)):
#     nrows, ncols = arr_ds.shape
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
#     fig.suptitle(f"{var} for each SOM node", fontsize=16)

#     for i in range(nrows):
#         for j in range(ncols):
#             ax = axes[i, j]
#             ds_patch = arr_ds[i, j]

#             if ds_patch is None or var not in ds_patch:
#                 ax.axis("off")
#                 continue

#             try:
#                 img = ds_patch[var].values
#                 if img.ndim == 3:
#                     # Assume (channel, y, x) → take visible or first channel
#                     img = img[0]  # Or use np.mean(img, axis=0) for multi-channel
#                 ax.imshow(img, cmap=cmap)
#             except Exception as e:
#                 ax.text(0.5, 0.5, "Error", ha="center", va="center", fontsize=6)
#                 ax.axis("off")
#                 continue

#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_title(f"({i},{j})", fontsize=6)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.95)
#     plt.show()