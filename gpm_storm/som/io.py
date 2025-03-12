#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:08:03 2023

@author: ghiggi
"""
import os
import logging
import numpy as np
import random
import dask
from gpm_storm.features.routines import get_gpm_storm_patch


def create_dask_cluster():
    import dask
    from dask.distributed import Client, LocalCluster

    # Set environment variable to avoid HDF locking
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # -------------------------------------------------------------------------.
    # Retrieve the number of process to run
    available_workers = os.cpu_count() - 2  # if not set, all CPUs
    num_workers = dask.config.get("num_workers", available_workers)

    # Create dask.distributed local cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit='100GB',
        silence_logs=logging.WARN,
    )

    Client(cluster)

    # Return the local cluster
    return cluster


def _get_node_dataframe(df, row, col):
    """Retrieve feature dataframe of specific SOM node."""
    df_node = df[(df['row'] == row) & (df['col'] == col)]
    return df_node


def create_som_df_features_stats(df): 
    # Groupby i,j  .apply mean/mean, max, min 
    grouped_df = df.groupby(['row', 'col'])
    
    # Calculate the mean for each group and each variable
    mean_df = grouped_df.mean()
    mean_df = grouped_df.mean().reset_index()
    return mean_df
    

def create_som_df_array(som, df):
    """Create SOM array with feature dataframe for each node."""
    som_shape = som.codebook.shape[:-1]
    arr_df = np.empty(som_shape, dtype=object)
    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            df_node = _get_node_dataframe(df, row=row, col=col)
            arr_df[row, col] = df_node
    return arr_df 


def _open_sample_dataset(df, index, variables="precipRateNearSurface"):
    granule_id = df.iloc[index]['gpm_granule_id']
    slice_start = df.iloc[index]['along_track_start']
    slice_end = df.iloc[index]['along_track_end']
    date = df.iloc[index]['time']
    ds = get_gpm_storm_patch(
        granule_id=granule_id,
        slice_start=slice_start,
        slice_end=slice_end,
        date=date,
        verbose=False,
        variables=variables,
    )
    return ds


def _open_dataset_patch(dict_query, variables="precipRateNearSurface"):

    ds = get_gpm_storm_patch(
        verbose=False,
        variables=variables,
        **dict_query
    )
    ds = ds.compute()
    return ds


@dask.delayed
def _delayed_open_dataset_patch(dict_query, variables): 
    with dask.config.set(scheduler="single-threaded"):
        ds = _open_dataset_patch(dict_query, variables=variables)
    return ds


def _get_sample_dict_query(df, index): 
    """Return dictionary with relevant info for extracting a GPM patch."""
    dict_query = {}
    dict_query["granule_id"] = df.iloc[index]['gpm_granule_id']
    dict_query["slice_start"] = df.iloc[index]['along_track_start']
    dict_query["slice_end"] = df.iloc[index]['along_track_end']
    dict_query["date"] = df.iloc[index]['time']
    return dict_query


def _get_sample_dict_queries(df, indices): 
    """Return dictionary with relevant info for extracting the GPM patches."""
    dict_queries = {index: _get_sample_dict_query(df, index) for index in indices}
    return dict_queries


def _get_som_nodes_queries_array(arr_df):
    """Return SOM array with queries information."""
    som_shape = arr_df.shape
    arr_queries = np.empty(som_shape, dtype=object)    
    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            # Extract SOM node dataframe 
            df_node = arr_df[row, col]
            # Select valid random index
            index = random.randint(0, len(df_node) - 1)
            # Get GPM patch query info 
            dict_info = _get_sample_dict_query(df_node, index)
            # Add the dataset to the arrays
            arr_queries[row, col] = dict_info
    print(arr_queries)
    return arr_queries


def _parallel_get_som_sample_ds_array(arr_df, variables):
    arr_queries = _get_som_nodes_queries_array(arr_df)
    arr_ds = np.empty(arr_queries.shape, dtype=object)
    list_indices = [] 
    list_delayed = []
    for row in range(arr_queries.shape[0]):
        for col in range(arr_queries.shape[1]):
            list_indices.append((row, col))
            list_delayed.append(_delayed_open_dataset_patch(arr_queries[row, col], variables)) 
    
    list_ds = dask.compute(*list_delayed)
    for (row, col), ds in zip(list_indices, list_ds):   
        arr_ds[row, col] = ds
    return arr_ds   
      
 
def _get_som_sample_ds_array(arr_df, variables):
    som_shape = arr_df.shape
    arr_ds = np.empty(som_shape, dtype=object)

    for row in range(som_shape[0]):
        for col in range(som_shape[1]):
            # Extract images for each cell in the SOM
            df_node = arr_df[row, col]
            # Select valid random index
            index = random.randint(0, len(df_node) - 1)
            # Open dataset
            ds = _open_sample_dataset(df_node, index=index, variables=variables)
            # Add the dataset to the arrays
            arr_ds[row, col] = ds
    return arr_ds

 
def create_som_sample_ds_array(arr_df, variables="precipRateNearSurface", parallel=True):
    """Random sample a single GPM patch for each SOM node."""
    if parallel: 
        return _parallel_get_som_sample_ds_array(arr_df, variables)
    else: 
        return _get_som_sample_ds_array(arr_df, variables)


def _get_node_datasets(df_node, num_images, variables):
    random_indices = random.sample(range(len(df_node)), num_images)
    list_ds = []
    for index in random_indices:
        print(index)
        ds = _open_sample_dataset(df_node, index=index, variables=variables)
        list_ds.append(ds)
    return list_ds


def _parallel_get_node_datasets(df_node, num_images, variables):
    random_indices = random.sample(range(len(df_node)), num_images)
    dict_queries = _get_sample_dict_queries(df_node, indices=random_indices)
    list_delayed = []
    for dict_query in dict_queries.values():
        list_delayed.append(_delayed_open_dataset_patch(dict_query, variables)) 
    list_ds = dask.compute(*list_delayed) 
    return list_ds


def sample_node_datasets(df_node, num_images=20, variables="precipRateNearSurface", parallel=True):
    if parallel: 
        return _parallel_get_node_datasets(df_node=df_node, num_images=num_images, variables=variables)
    else: 
        return _get_node_datasets(df_node=df_node, num_images=num_images, variables=variables)
    
