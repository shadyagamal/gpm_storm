import os 
import dask
import gpm
import numpy as np
import pandas as pd
import ximage  # noqa
import xarray as xr
from gpm.io.local import get_time_tree
from gpm.io.checks import check_date, check_time
from gpm.io.info import get_start_time_from_filepaths, get_granule_from_filepaths
from gpm.io.find import find_filepaths
from gpm_storm.gpm_storm.features.image import calculate_image_statistics
from datetime import timedelta



@dask.delayed
def create_gpm_storm_db(filepath, output_dir):
    try:
        with dask.config.set(scheduler="single-threaded"):
            compute_gpm_storm_db(filepath, output_dir)
    except Exception as e:
        error_str = str(e)
        msg = f"Error for {filepath}: {error_str}"
        return msg
    return None


def compute_gpm_storm_db(filepath, output_dir):
    """
    Process a single GPM file: Extract patches, compute statistics, and save results.
    
    Parameters:
    - filepath (str): Path to the GPM data file.
    
    Returns:
    - None
    """
 
    # Define variables to open
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

    # Load dataset
    ds = gpm.open_granule_dataset(filepath, 
                                  variables=variables, 
                                  scan_mode="FS",
                                  chunks={})
    
    # Label storms
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
    
    # Extract patches
    label_isel_dict = xr_obj.ximage.label_patches_isel_dicts(
        label_name="label",
        patch_size=(49, 49),
        variable="precipRateNearSurface",
        n_patches=300,
        n_labels=None,
        labels_id=None,
        padding=0,
        centered_on="max",
        partitioning_method=None,
        debug=False,
    )
    
    # Compute patch statistics
    patch_statistics = []
    stacked_patches = []
    
    granule_id = ds["gpm_granule_id"].data[0].item()
    gpm_id_start = ds["gpm_id"].isel(along_track=0).values.item()
    gpm_id_end = ds["gpm_id"].isel(along_track=-1).values.item()
    first_time = ds["time"].isel(along_track=0).values
    last_time = ds["time"].isel(along_track=-1).values
    
    for patch_id, isel_dict in label_isel_dict.items():
        ds_patch = ds.isel(**isel_dict[0]).compute()
        
        stats = calculate_image_statistics(ds_patch)
        stats["patch_id"] = patch_id-1
        patch_statistics.append(stats)
            
        # Stack patches along a new dimension
        ds_patch = ds_patch.expand_dims("patch", axis=0)
        for var in ["SCorientation", "dataQuality", "lon", "lat", "gpm_along_track_id", "height", "time", "gpm_id", "gpm_granule_id"]:
            ds_patch[var] = ds_patch[var].expand_dims("patch", axis=0)
        
        ds_patch = ds_patch.drop_vars(ds_patch.gpm.vertical_variables)
        ds_patch = ds_patch.drop_vars("height")
         
        stacked_patches.append(ds_patch)
    
    # Ensure output directory exists
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    relative_path = os.path.join(*filepath.split(os.sep)[-4:-2])
    parquet_save_dir = os.path.join(output_dir,"parquet/", relative_path)
    os.makedirs(parquet_save_dir, exist_ok=True)
    zarr_save_dir = os.path.join(output_dir,"zarr/", relative_path)
    os.makedirs(zarr_save_dir, exist_ok=True)

    # File names
    filename = os.path.basename(filepath).replace(".HDF5", "").replace(".", "_")
    parquet_path = os.path.join(parquet_save_dir, f"{filename}.parquet")
    zarr_path = os.path.join(zarr_save_dir, f"{filename}.zarr")
    
    # Save statistics as Parquet
    df = pd.DataFrame(patch_statistics)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S.%f")
    df.to_parquet(parquet_path)
    
    # Save patch images as Zarr
    if stacked_patches:
        ds_stacked = xr.concat(stacked_patches, dim="patch")
        
        ds_stacked["gpm_granule_id"] = granule_id
        ds_stacked.attrs["gpm_id_start"] = gpm_id_start
        ds_stacked.attrs["gpm_id_end"] = gpm_id_end
        ds_stacked.attrs["first_time"] = first_time
        ds_stacked.attrs["last_time"] = last_time
        ds_stacked.to_zarr(zarr_path, mode="w") 

    return None
        


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
    ds = gpm.open_granule(filepath, variables=variables, scan_mode="FS")
    
    # Put in memory data for label definition 
    ds["precipRateNearSurface"] = ds["precipRateNearSurface"].compute()
    da = ds["precipRateNearSurface"]
    
    # 
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
        
    # patch statistics extraction
        
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
    ds = gpm.open_granule(filepath, variables=variables, scan_mode=scan_mode, chunks=chunks)
    if (slice_end - slice_start < 49):
        slice_end =slice_start + 49
    ds = ds.isel(along_track=slice(slice_start, slice_end))
    return ds
