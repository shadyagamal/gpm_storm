#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:28:29 2023

@author: ghiggi
"""
"""
Spyder Editor

This is a temporary script file.
"""
import os 
import dask
import gpm_api
import numpy as np
import pandas as pd
import ximage  # noqa
from gpm_api.io.local import get_time_tree, get_local_daily_filepaths
from gpm_api.io.checks import check_date, check_time
from gpm_api.io.info import get_start_time_from_filepaths, get_granule_from_filepaths
from gpm_api.io.find import find_filepaths
from gpm_storm.features.image import calculate_image_statistics
from datetime import timedelta



@dask.delayed
def run_feature_extraction(filepath, dst_dir, force):
    with dask.config.set(scheduler="single-threaded"):
        try: 
            run_granule_feature_extraction(filepath, dst_dir=dst_dir, force=force)
            msg = ""
        except Exception as e: 
            msg = f"Processing of {filepath} failed with '{e}'."
    return msg 


def run_granule_feature_extraction(filepath, dst_dir, force=False):
    
    # Define filepath 
    start_time = get_start_time_from_filepaths(filepath)[0]
    filename = os.path.basename(filepath).replace(".HDF5", "")
    filename = f"GPM_STORM.{filename}.parquet"
    dirtree = get_time_tree(check_date(check_time(start_time)))
    dir_path = os.path.join(dst_dir, dirtree)
    os.makedirs(dir_path, exist_ok=True)
    df_filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(df_filepath): 
        if force: 
            os.remove(df_filepath)
        else: 
            raise ValueError(f"force=False and {filepath} already exists.")

    # List some variables of interest
    variables = [
        "sunLocalTime",
        "airTemperature",
        # "precipRate",
        # "paramDSD",
        "zFactorFinal",
        # "zFactorMeasured",
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
        "zFactorFinalESurface",
        "zFactorFinalNearSurface",
        "heightZeroDeg",
        "binEchoBottom",
        "landSurfaceType",
    ]
    
    # Open granule dataset
    ds = gpm_api.open_granule(filepath, variables=variables, scan_mode="FS")
    
    # Put in memory data for label definition 
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    da = ds["precipRateNearSurface"]
    
    # %%
    ###################
    #### Labelling ####
    ###################
    min_value_threshold = 0.05
    max_value_threshold = np.inf
    min_area_threshold = 5
    max_area_threshold = np.inf
    footprint = 5
    sort_by = "area"
    sort_decreasing = True
    label_name = "label"
  
    
    # Retrieve labeled xarray object
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
       
    ##################################
    #### Label Patches Extraction ####
    ##################################
    patch_size = (49, 20)
    variable = "precipRateNearSurface"
    # Output Options
    label_name = "label"
    labels_id = None
    n_labels = None
    n_patches = np.Inf
    # Patch Extraction Options
    centered_on = "label_bbox"
    padding = 0
    # Define the patch generator
    patch_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        # Output options
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        # Patch extraction Options
        padding=padding,
        centered_on=centered_on,
        # Tiling/Sliding Options
        partitioning_method=None,
    )
        
    # %% patch statistics extraction
        
    # Read first in memory to speed up computations [9 seconds]
    ds["airTemperature"] = ds["airTemperature"].compute()
    ds["zFactorFinal"] = ds["zFactorFinal"].compute()
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    ds["sunLocalTime"] = ds["sunLocalTime"].compute()
    
    # Compute statistics for each patch
    n_patches = len(patch_isel_dict)
    patch_statistics = [
        calculate_image_statistics(ds, patch_isel_dict[i][0]) for i in range(1, n_patches)
    ]
        
    # Create a pandas DataFrame from the list
    df = pd.DataFrame(patch_statistics)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f')
    # Save DataFrame to Parquet
    df.to_parquet(df_filepath)


def get_gpm_storm_patch(granule_id, 
                        slice_start, 
                        slice_end,
                        date, 
                        product="2A-DPR",
                        scan_mode="FS",
                        chunks={},
                        verbose=True,
                        variables=["precipRateNearSurface"]):
    
    start_time = date - timedelta(hours = 5)
    end_time = date + timedelta(hours = 5)
    
    filepaths = find_filepaths(product=product,  
                               product_type="RS", 
                               storage="local", 
                               version=7, 
                               start_time=start_time, 
                               end_time=end_time, 
                               verbose=verbose,
                               parallel=False,
                               )
    print(filepaths)
    if len(filepaths) == 0:
        raise ValueError(f"No file available between {start_time} and {end_time}")
    granule_ids = get_granule_from_filepaths(filepaths)
    indices = [i for i, iid in enumerate(granule_ids) if iid == granule_id]
    if len(indices) == 0: 
        raise ValueError(f"File corresponding to granule_id {granule_id} not found !")
    filepath = filepaths[indices[0]]
    if verbose:
        print(f"filepath: {filepath}")
        
    # Open granule dataset
    ds = gpm_api.open_granule(filepath, variables=variables, scan_mode=scan_mode, chunks=chunks)
    if (slice_end - slice_start < 49):
        slice_end =slice_start + 49
    ds = ds.isel(along_track=slice(slice_start, slice_end))
    return ds

