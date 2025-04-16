#!/usr/bin/env python3
"""

@author: shadya
"""
# %% Imports
import os
from pathlib import Path

import numpy as np
from PIL import Image

# %% Step 1: Define the paths
figs_path = Path("~/figs")
Data_path = Path("~data/")

# %%
# Initialize lists to store image data
image_data = []
image_filenames = []

# Iterate through the files in the directory
for filename in os.listdir(figs_path):
    if filename.endswith(".png"):
        # Open the image and convert it to a NumPy array
        image = Image.open(os.path.join(figs_path, filename))
        image_array = np.array(image)

        # Append the image data and filename to lists
        image_data.append(image_array)
        image_filenames.append(filename)

# %%
