"""
Created on Thu Dec 21 12:31:33 2023

@author: ghiggi
"""
import somoclu
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import sys
import os

## Import local function
PACKAGE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm_storm.gpm_storm.som.experiments import get_experiment_info, save_som
# from gpm_storm.gpm_storm.features.dataset_analysis import filter_nan_values # TO PUT IN gpm_storm.som.preprocessing !  

 
file_path = "~/gpm_storm/data/patch_statistics.parquet"
som_dir = "~/gpm_storm/script" # TODO to change ... 
som_name = "high_intensity_SOM" # TODO: THIS IS THE NAME IDENTIFYING THE EXPERIMENT
    
#--------------------------------------------------------------------------------.
# Read the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Define the features to train the SOM 
info_dict = get_experiment_info(som_name)  # HERE INSIDE YOU DEFINE THE EXPERIMENT (features, som_settings ...)
features = info_dict["features"]
n_rows, n_columns = info_dict["som_grid_size"] 

# Subset here the dataframe row to discard (i.e. nan stuffs, select only high intensity ...)
# TODO: TO IMPROVE ! IDEALLY PARAMETRIZE IN get_experiment_info !
# How can you recall the preprocessing ... if you modify manually the code for each trained som !!!! 
# # for feature in features:
#     df = filter_nan_values(df, features)
df = df.dropna(subset=features) # MAYBE THIS IS ENOUGH for the moment ... but are discarding lot of stuffs... to be reported ! 
 
# Select columns with the relevant dataframe features
df_features = df[features]

# Standardize the dataframe 
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features))

# Extract the data array
data = df_scaled.to_numpy()

# Initialize the SOM
som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, \
                      gridtype='rectangular', maptype='planar') #  initialcodebook)  ...sample the original codes 

# Train SOM
# train(data=None, epochs=10,  scale0=0.1, scaleN=0.001, scalecooling='linear')
som.train(data=data, epochs=100, \
          radius0=0, radiusN=1, \
          scale0=0.5, scaleN=0.001)

# Save the trained SOM
save_som(som, som_dir=som_dir, som_name=som_name)