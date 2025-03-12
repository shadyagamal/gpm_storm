#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:24:28 2023

@author: ghiggi
"""
import pickle
import os 
# TODO: HERE YOU DESCRIBE THE FEATURE YOU USE
# the name is the experiment som_name 

def get_experiment_info(name): 
    info_experiment = {
        "zonal_SOM": {
            "features": ['precipitation_average', 'precipitation_std', 'precipitation_pixel',
                   'center_precipitation_pixel', 'precipitation_sum', 'precipitation_max','count_rainy_areas_over_0', 'mean_for_rainy_pixels_over_0',
                   'count_rainy_pixels_over_0', 'major_axis_largest_patch_over_0',
                   'minor_axis_largest_patch_over_0', 'aspect_ratio_largest_patch_over_0',
                   'count_rainy_pixels_in_patch_over_0',  'percentage_rainy_pixels_between_0_and_1',
                    'percentage_rainy_pixels_between_1_and_2',
                    'percentage_rainy_pixels_between_2_and_5',
                    'percentage_rainy_pixels_between_5_and_10',
                    'percentage_rainy_pixels_between_10_and_20','REFCH_mean','REFCH_max', 'echodepth18_mean', 'echodepth18_max'],
            "som_grid_size": (10, 10)
           },
        "shape_SOM": {
            "features": ['precipitation_average', 'precipitation_std', 'precipitation_pixel',
                   'center_precipitation_pixel', 'precipitation_sum', 'precipitation_max',
                   'count_rainy_areas_over_0', 'mean_for_rainy_pixels_over_0',
                   'count_rainy_pixels_over_0', 'major_axis_largest_patch_over_0',
                   'minor_axis_largest_patch_over_0', 'aspect_ratio_largest_patch_over_0',
                   'count_rainy_pixels_in_patch_over_0', 'count_rainy_areas_over_1',
                   'mean_for_rainy_pixels_over_1', 'count_rainy_pixels_over_1',
                   'major_axis_largest_patch_over_1', 'minor_axis_largest_patch_over_1',
                   'aspect_ratio_largest_patch_over_1',
                   'count_rainy_pixels_in_patch_over_1', 'count_rainy_areas_over_2',
                   'mean_for_rainy_pixels_over_2', 'count_rainy_pixels_over_2',
                   'major_axis_largest_patch_over_2', 'minor_axis_largest_patch_over_2',
                   'aspect_ratio_largest_patch_over_2',
                   'count_rainy_pixels_in_patch_over_2', 'count_rainy_areas_over_5',
                   'mean_for_rainy_pixels_over_5', 'count_rainy_pixels_over_5',
                   'major_axis_largest_patch_over_5', 'minor_axis_largest_patch_over_5',
                   'aspect_ratio_largest_patch_over_5',
                   'count_rainy_pixels_in_patch_over_5', 'count_rainy_areas_over_10',
                   'mean_for_rainy_pixels_over_10', 'count_rainy_pixels_over_10',
                   'major_axis_largest_patch_over_10', 'minor_axis_largest_patch_over_10',
                   'aspect_ratio_largest_patch_over_10',
                   'count_rainy_pixels_in_patch_over_10',
                   'percentage_rainy_pixels_between_0_and_1',
                   'percentage_rainy_pixels_between_1_and_2',
                   'percentage_rainy_pixels_between_2_and_5',
                   'percentage_rainy_pixels_between_5_and_10',
                   'percentage_rainy_pixels_between_10_and_20'],
                "som_grid_size": (10, 10)
            },
        "shape_SOM1": {
            "features": ['precipitation_average', 'precipitation_std', 'precipitation_pixel',
                   'center_precipitation_pixel', 'precipitation_sum', 'precipitation_max',
                   'count_rainy_areas_over_0', 'mean_for_rainy_pixels_over_0',
                   'count_rainy_pixels_over_0', 'major_axis_largest_patch_over_0',
                   'minor_axis_largest_patch_over_0', 'aspect_ratio_largest_patch_over_0',
                   'count_rainy_pixels_in_patch_over_0', 'count_rainy_areas_over_1',
                   'mean_for_rainy_pixels_over_1', 'count_rainy_pixels_over_1',
                   'major_axis_largest_patch_over_1', 'minor_axis_largest_patch_over_1',
                   'aspect_ratio_largest_patch_over_1',
                   'count_rainy_pixels_in_patch_over_1', 'count_rainy_areas_over_2',
                   'mean_for_rainy_pixels_over_2', 'count_rainy_pixels_over_2',
                   'major_axis_largest_patch_over_2', 'minor_axis_largest_patch_over_2',
                   'aspect_ratio_largest_patch_over_2',
                   'count_rainy_pixels_in_patch_over_2', 'count_rainy_areas_over_5',
                   'mean_for_rainy_pixels_over_5', 'count_rainy_pixels_over_5',
                   'major_axis_largest_patch_over_5', 'minor_axis_largest_patch_over_5',
                   'aspect_ratio_largest_patch_over_5',
                   'count_rainy_pixels_in_patch_over_5', 'count_rainy_areas_over_10',
                   'mean_for_rainy_pixels_over_10', 'count_rainy_pixels_over_10',
                   'major_axis_largest_patch_over_10', 'minor_axis_largest_patch_over_10',
                   'aspect_ratio_largest_patch_over_10',
                   'count_rainy_pixels_in_patch_over_10',
                   'percentage_rainy_pixels_between_0_and_1',
                   'percentage_rainy_pixels_between_1_and_2',
                   'percentage_rainy_pixels_between_2_and_5',
                   'percentage_rainy_pixels_between_5_and_10',
                   'percentage_rainy_pixels_between_10_and_20'],
                "som_grid_size": (10, 10)
            }

    }
    return info_experiment[name]


def filter_nan_values(df, feature):
    """ Filter out rows where a specified variable has NaN values."""
    filtered_df = df.dropna(subset=[feature])
    return filtered_df



def save_som(som, som_dir, som_name):
    # Save SOM
    som_filepath = os.path.join(som_dir, som_name + ".pkl")

    # Save the trained SOM
    with open(som_filepath, 'wb') as file:
        pickle.dump(som, file)
       
        
def load_som(som_dir, som_name):  
    som_filepath = os.path.join(som_dir, som_name + ".pkl")

    with open(som_filepath, 'rb') as file:
        som = pickle.load(file)  
    return som 

