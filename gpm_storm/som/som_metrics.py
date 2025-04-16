#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:35:33 2025

@author: gamal
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def quantization_error_old(som, data):
    """Calculate the Quantization Error (QE)"""
    # Compute the distance between the data points and the corresponding map units
    distances = cdist(data, som.get_codebook().reshape(-1, som.get_codebook().shape[-1]))
    min_distances = np.min(distances, axis=1)
    qe = np.mean(min_distances)
    return qe


def topographic_product_old(som, data):
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


def topographic_error_old(som, data):
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


def trustworthiness_old(som, data):
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


def neighborhood_preservation_old(som, data):
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

def trustworthiness(X, W, bmus, n_neighbors=5):
    """
    Computes trustworthiness of SOM mapping.
    """
    n_samples = X.shape[0]
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    X_neighbors = knn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    som_codes = W[bmus[:, 0], bmus[:, 1]]
    som_knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(som_codes)
    som_neighbors = som_knn.kneighbors(som_codes, return_distance=False)[:, 1:]

    trust_sum = 0
    for i in range(n_samples):
        missing = np.setdiff1d(X_neighbors[i], som_neighbors[i], assume_unique=True)
        for rank, j in enumerate(missing):
            trust_sum += rank + 1  # ranks start from 1

    u = 1 - (2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))) * trust_sum
    return u

def neighborhood_preservation(X, W, bmus, n_neighbors=5):
    """
    Computes neighborhood preservation: average overlap of neighbors.
    """
    n_samples = X.shape[0]
    knn_input = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    input_neighbors = knn_input.kneighbors(X, return_distance=False)[:, 1:]

    som_codes = W[bmus[:, 0], bmus[:, 1]]
    knn_som = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(som_codes)
    som_neighbors = knn_som.kneighbors(som_codes, return_distance=False)[:, 1:]

    overlaps = []
    for i in range(n_samples):
        overlap = len(np.intersect1d(input_neighbors[i], som_neighbors[i]))
        overlaps.append(overlap / n_neighbors)

    return np.mean(overlaps)

def quantization_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    distances = cdist(X, flat_W)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)

def topographic_error(W, X):
    flat_W = W.reshape(-1, W.shape[-1])
    codesize = W.shape[:2]
    errors = []

    for x in X:
        dists = np.linalg.norm(flat_W - x, axis=1)
        bmu_indices = np.argsort(dists)[:2]
        bmu1 = np.unravel_index(bmu_indices[0], codesize)
        bmu2 = np.unravel_index(bmu_indices[1], codesize)
        dist = np.linalg.norm(np.array(bmu1) - np.array(bmu2))
        errors.append(0 if dist == 1 else 1)

    return np.mean(errors)

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