#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:07:29 2025

@author: gamal
"""

import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ximage  # noqa

import gpm
from gpm.utils.geospatial import (
    get_circle_coordinates_around_point,
    get_country_extent,
    get_geographic_extent_around_point,
)

import gpm
import datetime
from matplotlib.colors import LogNorm, Normalize
from xhistogram.xarray import histogram

sample = node_df.iloc[90]

_, patch_ds = find_zarr_file_for_patch(sample, zarr_directory)
patch_ds.gpm.variables
patch_ds["typePrecip"]


ds = gpm.open_dataset(
    product="2A-DPR",
    product_type="RS",
    version=7,
    start_time=sample["time"] - datetime.timedelta(minutes=10),
    end_time=sample["time"] + datetime.timedelta(minutes=10),
    scan_mode="FS"
)

start_gpm_id = sample["gpm_id_start"]
end_gpm_id = sample["gpm_id_end"]
patch_ds = ds.gpm.sel(gpm_id=slice(start_gpm_id, end_gpm_id))
patch_ds["gpm_id"].data

z_ku = patch_ds["zFactorFinal"].sel(radar_frequency="Ku")


type_precip = patch_ds["typePrecip"]  # (cross_track, along_track)

# Stack to 1D for easier filtering
ds_patch = patch_ds.stack(footprint=["cross_track", "along_track"])
z_ku = ds_patch["zFactorFinal"].sel(radar_frequency="Ku")  # (footprint, range)
type_precip = ds_patch["typePrecip"]  # (footprint)

# Filter for convective or stratiform footprints
z_conv = z_ku.sel(footprint=type_precip == 1)
z_strat = z_ku.sel(footprint=type_precip.isin([2, 3]))

# CFAD histograms as before
reflectivity_bins = np.arange(10, 60, 0.5)
z_conv_dens = histogram(z_conv, bins=reflectivity_bins, dim=["footprint"])
z_strat_dens = histogram(z_strat, bins=reflectivity_bins, dim=["footprint"])

# Plot
z_conv_dens.plot.imshow(x="zFactorFinal_bin", y="range", origin="upper", cmap="Spectral_r")
plt.title("Convective CFAD")
plt.show()

z_strat_dens.plot.imshow(x="zFactorFinal_bin", y="range", origin="upper", cmap="Spectral_r")
plt.title("Stratiform CFAD")
plt.show()

def retrieve_patch_data(sample, product="2A-DPR", version=7, product_type="RS"):
    start = sample["time"] - datetime.timedelta(minutes=10)
    end = sample["time"] + datetime.timedelta(minutes=10)

    ds = gpm.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start,
        end_time=end,
        variables = ["zFactorFinal"],
        scan_mode="FS"
    )
    start_gpm_id = sample["gpm_id_start"]
    end_gpm_id = sample["gpm_id_end"]

    ds_patch = ds.gpm.sel(gpm_id=slice(start_gpm_id, end_gpm_id))
    z_ku = ds_patch["zFactorFinal"].sel(radar_frequency="Ku")

    return z_ku, ds_patch