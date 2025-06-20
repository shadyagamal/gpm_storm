import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from xhistogram.xarray import histogram
from matplotlib.colors import Normalize
import gpm
import os
import matplotlib.patches as patches
import pandas as pd
import pickle
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure

# --- Load Data ------
som_name = "SOM_Pmean_>_1_with_random_init"  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
figs_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/CFAD")
res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/results")
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
df_bmu = pd.read_parquet(bmu_dir)

actual_combinations = set(zip(df_bmu["row"], df_bmu["col"], strict=False))

def compute_wcc_flags_3d(z_volume, threshold, area_thresh_km2, pixel_area_km2=25):
    binary_mask = z_volume > threshold
    structure3d = generate_binary_structure(3, 2) 
    structure2d = generate_binary_structure(2, 2) 
    labeled_3d, num_features = label(binary_mask, structure=structure3d) 
    

    for i in range(1, num_features + 1):
        obj_mask = (labeled_3d == i)

        for z in range(obj_mask.shape[0]): 
            horizontal_slice = obj_mask[z]

            labeled_2d, num_2d = label(horizontal_slice, structure=structure2d)

            for j in range(1, num_2d + 1):
                region = (labeled_2d == j)
                area_km2 = region.sum() * pixel_area_km2
                if area_km2 >= area_thresh_km2:
                    return True 
    return False

def compute_wcc_flag(node_df, top_pixels=None):
    wcc_flags = []
    
    for i in tqdm(range(len(node_df)//3)):
        sample = node_df.iloc[i]
        try:
            ds = gpm.open_dataset(
                product="2A-DPR",
                product_type="RS",
                version=7,
                start_time=sample["time"] - datetime.timedelta(minutes=1),
                end_time=sample["time"] + datetime.timedelta(minutes=1),
                scan_mode="FS",
                chunks={},
                variables=["zFactorFinal", "height", "typePrecip", "heightZeroDeg"]
                )
            ds_patch = ds.gpm.sel(gpm_id=slice(sample["gpm_id_start"], sample["gpm_id_end"])).compute()
            ds_patch_footprints = ds_patch.stack(footprints=["cross_track", "along_track"])
         
            z_ku = ds_patch_footprints["zFactorFinal"].sel(radar_frequency="Ku")
            z_cube = z_ku.unstack("footprints").transpose("range", "cross_track", "along_track").values

            wcc_moderate = compute_wcc_flags_3d(z_cube, threshold=30, area_thresh_km2=800)
            wcc_strong = compute_wcc_flags_3d(z_cube, threshold=40, area_thresh_km2=1000)
            
            wcc_flags.append({
                "index": sample.name,
                "wcc_moderate": wcc_moderate,
                "wcc_strong": wcc_strong
            })
            
                
        except Exception as e:
            print(e)
            continue
    return wcc_flags


# --- Main script ---

wcc_flagss = {}

for node in actual_combinations:
    row, col = node
    node_df = df_bmu[(df_bmu["row"] == row) & (df_bmu["col"] == col)].copy()
    print(f"{len(node_df)} events in node {node}")
    
    wcc_flags = compute_wcc_flag(node_df, top_pixels=5)
    wcc_flagss[row,col] = wcc_flags
    
with open(os.path.join(res_dir,"wcc_flagss.pkl"), "wb") as f:
    pickle.dump(wcc_flagss, f)