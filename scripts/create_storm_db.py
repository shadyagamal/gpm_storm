import os 
import gpm # noqa
import dask 
import logging
from gpm.io.local import get_local_filepaths
from gpm_storm.features.routines import create_gpm_storm_db
import zarr 
import numcodecs
import time
import xarray as xr
import pandas as pd


zarr.blosc.use_threads = False
numcodecs.blosc.use_threads = False

dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
dask.config.set({'distributed.worker.use-file-locking': 'False'})
dask.config.set({'logging.distributed': 'error'})

from dask.distributed import Client, LocalCluster


if __name__ == "__main__": #  https://github.com/dask/distributed/issues/2520
    # Notes 
    # - This code is penalized by HDF/netCDF locking 
    #   - Even if HDF5 is compiled with thread safety, the netcdf4 C library is not thread safe.
    #   - We use multiprocessing to get partially around that 
    # - This code use dask.delayed. dask.delayed works only with dask.distributed ! 
    
    ####----------------------------------------------------------------------.
    #### Define Dask Distributed Cluster   
    # Set environment variable to avoid HDF locking
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Set number of workers 
    # dask.delayed run n_workers*2 concurrent processes
    # available_workers = int(os.cpu_count()/2)
    # num_workers = dask.config.get("num_workers", available_workers)
        
    # Create dask.distributed local cluster
    # --> Use multiprocessing to avoid netCDF multithreading locks ! 
    cluster = LocalCluster(
        n_workers=20,
        threads_per_worker=1, # important to set to 1 to avoid netcdf locking ! 
        processes=True,
    )
    
    client = Client(cluster)
    
    ####----------------------------------------------------------------------.

    # set ulimit -n 999999
    
    base_dir = None 
    base_dir = "/ltenas2/data/GPM"
    # List output directory
    # output_dir="/t5500/ltenas2/data/GPM_STORM_DB"
    output_dir="/ltenas2/data/GPM_STORM_DB"
    
    # List available files
    filepaths = get_local_filepaths(base_dir=base_dir, product="2A-DPR", version=7, product_type="RS")
    filepaths = filepaths[0:200_000]
    
    
    # Define computations
    list_delayed = [create_gpm_storm_db(filepath=filepath, output_dir=output_dir)
                    for filepath in filepaths]
    # Run computations
    start_time = time.time()
    list_outputs = dask.compute(*list_delayed)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Time taken: {processing_time:.2f} seconds")
    # Report errors
    list_errors = [out for out in list_outputs if out is not None]
    for err in list_errors:
        print(err)

# for filepath in FILEPATH_LIST:
#     compute_gpm_storm_db(filepath, output_dir=OUTPUT_DIR)
#     print(f"Processed: {filepath}")

# compute_gpm_storm_db(FILEPATH_LIST[100], output_dir=OUTPUT_DIR)
# compute_gpm_storm_db(filepath, output_dir=output_dir)



# filepath = filepaths[0]
# output_dir = os.path.expanduser(output_dir)
# os.makedirs(output_dir, exist_ok=True)

# relative_path = os.path.join(*filepath.split(os.sep)[-4:-2])
# parquet_save_dir = os.path.join(output_dir,"parquet/", relative_path)
# os.makedirs(parquet_save_dir, exist_ok=True)
# zarr_save_dir = os.path.join(output_dir,"zarr/", relative_path)
# os.makedirs(zarr_save_dir, exist_ok=True)

# # File names
# filename = os.path.basename(filepath).replace(".HDF5", "").replace(".", "_")
# parquet_path = os.path.join(parquet_save_dir, f"{filename}.parquet")
# zarr_path = os.path.join(zarr_save_dir, f"{filename}.zarr")

# df = pd.read_parquet(parquet_path)
# print(df.head())
# print(df.info())


# ds_stacked = xr.open_zarr(zarr_path)
# print(ds_stacked)
# print(ds_stacked.variables)
# ds_stacked["precipRateNearSurface"].isel(patch=0).gpm.plot_image()
