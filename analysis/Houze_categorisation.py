import numpy as np
import os
import pandas as pd
import xarray as xr
import glob
from tqdm import tqdm


som_name = "SOM_Pmean_>_1_with_random_init" 

res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/0_Results") 
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
zarr_dir = "/ltenas2/data/GPM_STORM_DB/zarr"
flag_path = os.path.join(res_dir, "houze_flags.parquet")
df_bmu = pd.read_parquet(bmu_dir)


# --- Houze's Classification ---
def find_zarr_file_for_patch(row, zarr_directory, filename_pattern="*.zarr"):
    granule_id, patch_id = str(row["gpm_granule_id"]), row["patch_id"]
    time = pd.to_datetime(row["time"])
    
    for offset in [0, -1]:
        t = time + pd.DateOffset(months=offset)
        path = os.path.join(zarr_directory, f"{t.year:04d}/{t.month:02d}", filename_pattern)
        for zarr_file in glob.glob(path):
            if granule_id in os.path.basename(zarr_file):
                ds = xr.open_zarr(zarr_file)
                if patch_id < ds.sizes["patch"]:
                    return zarr_file, ds.isel(patch=patch_id)
    print(f"No Zarr for granule_id {granule_id} at {time.strftime('%Y-%m')}")
    return None, None

def classify_houze_categories(sample, zarr_directory):
    try:
        _, patch_ds = find_zarr_file_for_patch(sample, zarr_directory)
    except:
        return {
            "ISE": False,
            "DCC_moderate": False,
            "DCC_strong": False,
            "BSR_moderate": False,
            "BSR_strong": False,
            "H0":np.nan,
            "hail_flag_30": False,
            "hail_flag_40": False,
            "hail_flag_50": False,
            "hail_depth_30": np.nan,
            "hail_depth_40": np.nan,
            "hail_depth_50": np.nan
        }
    h0 = patch_ds["heightZeroDeg"].values.mean()
    precip_type = patch_ds.gpm.retrieve("flagPrecipitationType", method="major_rain_type")
    
    # ISE
    ise_flag = sample["ETH30_max"] < (h0 - 1000)

    # DCC
    dcc_moderate_flag = sample["ETH30_max"] > 8000
    dcc_strong_flag = sample["ETH40_max"] > 10000

    # BSR
    strat_mask = precip_type == 1  # 1 = stratiform
    strat_area = strat_mask.values.sum() * 25
    bsr_moderate = strat_area >= 40000
    bsr_strong = strat_area >= 50000

    # Hail
    hail_flag_30 = (sample["ETH30_max"] - h0) > 3000
    hail_depth_30 = sample["ETH30_max"] - h0
    hail_flag_40 = (sample["ETH40_max"] - h0) > 3000
    hail_depth_40 = sample["ETH40_max"] - h0
    hail_flag_50 = (sample["ETH50_max"] - h0) > 3000
    hail_depth_50 = sample["ETH50_max"] - h0

    return {
        "ISE": ise_flag,
        "DCC_moderate": dcc_moderate_flag,
        "DCC_strong": dcc_strong_flag,
        "BSR_moderate": bsr_moderate,
        "BSR_strong": bsr_strong,
        "H0": h0,
        "hail_flag_30": hail_flag_30,
        "hail_flag_40": hail_flag_40,
        "hail_flag_50": hail_flag_50,
        "hail_depth_30": hail_depth_30,
        "hail_depth_40": hail_depth_40,
        "hail_depth_50": hail_depth_50
    }



houze_flags = []
for i, sample in tqdm(df_bmu.iterrows()):
    flags = classify_houze_categories(sample, zarr_dir)
    houze_flags.append(flags)
houze_df = pd.DataFrame(houze_flags, index=df_bmu.index)


# houze_df.to_parquet(flag_path, index=True)
# houze_df = pd.read_parquet(flag_path)




