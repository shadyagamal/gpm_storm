#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:53:55 2025

@author: gamal
"""

#%% IMPORTS
import sys
import os
import datetime
import gpm # type: ignore
import numpy as np 
import pandas as pd 
import ximage  # noqa 
from matplotlib import pyplot as plt 
from gpm.io.local import get_local_filepaths # type: ignore

## Import local function
PACKAGE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm_storm.gpm_storm.features.image import calculate_image_statistics # type: ignore

#%%  Download data
### Time period and product
start_time = datetime.datetime.strptime("2023-08-20 20:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2023-08-22 00:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-DPR"  # GPM DPR Level 2A product (Dual-frequency Precipitation Radar)
product_type = "RS"
version = 7

gpm.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=False,
    verbose=True,
    progress_bar=True,
    check_integrity=False,
)

#%%  Chose variables of interest
variables = [
    "sunLocalTime",
    "airTemperature",
    "precipRate",
    "paramDSD",
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

#%%  Load data files 
#If working with all of the locally downloaded files working with the granule 
""" filepath_list = get_local_filepaths(product, version=version, product_type=product_type)
filepath_2023 = [fp for fp in filepath_list if "/2023/" in fp]

for filepath in filepath_2023:
    ds = gpm.open_granule(filepath, variables=variables, scan_mode="FS") """

ds = gpm.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        variables=variables,
        prefix_group=False,
    )

#%%  Storm labeling
min_value_threshold = 0.05
max_value_threshold = np.inf
min_area_threshold = 5
max_area_threshold = np.inf
footprint = 5
sort_by = "area"
sort_decreasing = True
label_name = "label"
da = ds["precipRateNearSurface"].compute()

xr_obj = da.ximage.label(
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
    label_name=label_name,
)
# Plot full label array
"""xr_obj[label_name].ximage.plot_labels()"""


#%%  Patch extraction
patch_size = (49, 20)
n_patches = 50
label_name = "label"
highlight_label_id = False
labels_id = None
n_labels = None
centered_on = "label_bbox"
padding = 0
variable = "precipRateNearSurface"

# Define the patch generator
label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
    label_name=label_name,
    patch_size=patch_size,
    variable=variable,
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    padding=padding,
    centered_on=centered_on,
    partitioning_method=None,
    debug=False,
)


#%%  Patch statistics
ds["zFactorFinal"] = ds["zFactorFinal"].compute()
ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
ds["sunLocalTime"] = ds["sunLocalTime"].compute()

patch_statistics = [
    calculate_image_statistics(ds, label_isel_dict[i][0]) for i in range(1, n_patches)
]

#%%  Save results as parquet
file_path = os.path.join(OUTPUT_DIR, "patch_statistics")
df = pd.DataFrame(patch_statistics)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f')
df.to_parquet(file_path)

print(f"Saved features to: {file_path}")