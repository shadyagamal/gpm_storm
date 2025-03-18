"""
Train a Self-Organizing Map (SOM) using patch statistics.

@author: shadya
"""

import os
import sys
import pandas as pd
import somoclu
from sklearn.preprocessing import MinMaxScaler 

## Import local function
PACKAGE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from gpm_storm.gpm_storm.som.experiments import get_experiment_info, save_som
# from gpm_storm.gpm_storm.features.dataset_analysis import filter_nan_values # TO PUT IN gpm_storm.som.preprocessing !  
 
FILEPATH = os.path.expanduser("~/gpm_storm/data/patch_statistics.parquet")  # f"feature_{granule_id}.parquet"
SOM_DIR = os.path.expanduser("~/gpm_storm/script") # TODO to change ... 
SOM_NAME = "zonal_SOM" # TODO: THIS IS THE NAME IDENTIFYING THE EXPERIMENT

def preprocess_data(df, features):
    """Preprocess dataset by filtering NaNs and normalizing."""
    print("🔍 Preprocessing data...")

    # Drop rows with NaN values in the selected features
    df_cleaned = df.dropna(subset=features)

    # Normalize using Min-Max scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cleaned[features]),
        columns=features,
        index=df_cleaned.index
    )
    print(f"Data preprocessing complete! {len(df) - len(df_cleaned)} rows removed due to NaNs.\n")
    return df_scaled

    
def train_som(df_scaled, som_name):
    """
    Train a Self-Organizing Map (SOM) using the selected features.
    """
    print(f"Training SOM for experiment: {som_name}")

    # Get experiment settings
    info_dict = get_experiment_info(som_name)  
    n_rows, n_columns = info_dict["som_grid_size"]  

    # Convert DataFrame to NumPy array
    data = df_scaled.to_numpy()

    # Initialize SOM
    som = somoclu.Somoclu(
        n_columns=n_columns, 
        n_rows=n_rows, 
        gridtype="rectangular", 
        maptype="planar"
    )

    # Train SOM
    som.train(
        data=data, 
        epochs=100,
        radius0=0, radiusN=1,
        scale0=0.5, scaleN=0.001
    )

    # Save the trained SOM
    save_som(som, som_dir=SOM_DIR, som_name=som_name)
    print("SOM training complete and model saved!\n")

def main():
    df = pd.read_parquet(FILEPATH)

    # Get experiment features
    experiment_info = get_experiment_info(SOM_NAME)
    features = experiment_info["features"]

    df_scaled = preprocess_data(df, features)

    train_som(df_scaled, SOM_NAME)

if __name__ == "__main__":
    main()