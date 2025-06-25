#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:35:33 2025

@author: gamal
"""

import numpy as np
from scipy.spatial.distance import cdist


def quantization_error(som, data):
    """Calculate the Quantization Error (QE)"""
    # Compute the distance between the data points and the corresponding map units
    distances = cdist(data, som.get_codebook().reshape(-1, som.get_codebook().shape[-1]))
    min_distances = np.min(distances, axis=1)
    qe = np.mean(min_distances)
    return qe


def topographic_product(som, data):
    """Calculate the Topographic Product"""
    topographic_errors = []
    for sample in data:
        best_matching_unit, second_best_matching_unit = som.get_bmus(sample)
        # Get the positions of the BMUs in the map grid
        bmu_position = np.unravel_index(best_matching_unit, som.get_codesize())
        second_bmu_position = np.unravel_index(second_best_matching_unit, som.get_codesize())
        # Calculate the distance in map grid space
        grid_distance = np.linalg.norm(np.array(bmu_position) - np.array(second_bmu_position))
        
        # Compute Topographic Product (P)
        p1 = grid_distance
        p2 = np.linalg.norm(sample - best_matching_unit)
        p3 = np.sqrt(p1 * p2)
        topographic_errors.append(p3)
    
    return np.mean(topographic_errors)


def topographic_error(som, data):
    """Calculate the Topographic Error"""
    errors = []
    for sample in data:
        best_matching_unit, second_best_matching_unit = som.get_bmus(sample)
        # Check if the best and second-best BMUs are adjacent on the map
        bmu_position = np.unravel_index(best_matching_unit, som.get_codesize())
        second_bmu_position = np.unravel_index(second_best_matching_unit, som.get_codesize())
        if np.linalg.norm(np.array(bmu_position) - np.array(second_bmu_position)) != 1:
            errors.append(1)
        else:
            errors.append(0)
    return np.mean(errors)


def trustworthiness(som, data):
    """Calculate the Trustworthiness metric"""
    errors = []
    for sample in data:
        best_matching_unit = som.get_bmus(sample)
        # Compare the closest neighbors in the original space to the best matching units
        neighbors_in_input = find_closest_neighbors(sample, data)
        neighbors_in_output = find_nearest_neighbors_on_map(sample, best_matching_unit)
        
        # Count errors when the closest neighbors in input space do not match output space neighbors
        errors.extend([1 if n1 != n2 else 0 for n1, n2 in zip(neighbors_in_input, neighbors_in_output)])

    return 1 - np.mean(errors)


def neighborhood_preservation(som, data):
    """Calculate the Neighborhood Preservation metric"""
    errors = []
    for sample in data:
        best_matching_unit = som.get_bmus(sample)
        neighbors_in_output = find_nearest_neighbors_on_map(sample, best_matching_unit)
        
        # Compare the closest neighbors in output space to the neighbors in input space
        neighbors_in_input = find_closest_neighbors(sample, data)
        
        # Count errors when the output space neighbors are not aligned with input space neighbors
        errors.extend([1 if n1 != n2 else 0 for n1, n2 in zip(neighbors_in_input, neighbors_in_output)])

    return 1 - np.mean(errors)


def find_closest_neighbors(sample, data):
    """Find closest neighbors in input space"""
    distances = cdist([sample], data)
    closest_neighbors = np.argsort(distances[0])[:5]  # Select the top 5 closest neighbors
    return closest_neighbors


def find_nearest_neighbors_on_map(sample, bmu):
    """Find the nearest neighbors on the map grid"""
    neighbors = []
    # Implement a method to find the neighbors of the BMU on the map (based on your SOM's grid)
    return neighbors