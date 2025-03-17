#!/usr/bin/env python3
"""

@author: shadya
"""
#%% Imports
from pathlib import Path
from netCDF4 import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

import umap
import somoclu

#%% Step 1: Define the paths
figs_path = Path("~/gpm_storm/figs")
Data_path = Path("~/gpm_storm/data/")

# %%
