import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import pickle



def plot_10x10_cfad(cfads, cfad_path, precip_type):
    nrows, ncols = 10, 10 
    figsize = (15, 15)
    reflectivity_ticks = np.arange(30, 70, 10)
    relative_height_ticks = np.arange(-7000, 10000, 3000)
    zmin, zmax = 10, 60
    hmin, hmax = -7000, 10000
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    for row, col in actual_combinations:
        i, j = row, col
        ax = axes[i, j]
        cfad = cfads[i, j]

        cfad.where(cfad > 0).plot.imshow(
            ax=ax, x="zFactorFinal_bin", y="relheight_bin", origin="lower",
            cmap="Spectral_r", add_colorbar=False,  add_labels=False)
        ax.set_title(f"({i},{j})", fontsize=10)
        ax.set_xticks(reflectivity_ticks)
        ax.set_yticks(relative_height_ticks)
        ax.set_xlim(zmin, zmax)
        ax.set_ylim(hmin, hmax)
        ax.tick_params(axis="both", which="both", labelsize=6)
        ax.tick_params(which="minor", left=False)
        ax.grid()
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, edgecolor='black', facecolor='none'))
    
    
    fig.suptitle(f"{precip_type} CFADs by Node", fontsize=16)
    fig.tight_layout()
    # cfad_path = os.path.join(figs_dir, f"{precip_type}_CFADs_10x10.png")
    # fig.savefig(cfad_path, dpi=300)
    # plt.close(fig)
    plt.show(fig)
    
    
def individual_CFAD(cfads, node, precip_type):
    cfad = cfads[node[0],node[1]]
    plt.figure(figsize=(6, 5))
    cfad.where(cfad > 0).plot.imshow(
        x="zFactorFinal_bin", y="relheight_bin", origin="lower",
        cmap="Spectral_r", extend="both")
    plt.title(f"{precip_type} CFAD: node({node[0]},{node[1]})")
    plt.show()
    
# --- Load Data ------
som_name = "SOM_Pmean_>_1_with_random_init"  
som_dir = os.path.expanduser("~/gpm_storm/data/trained_soms/")  
bmu_dir = os.path.expanduser(f"~/gpm_storm/data/df_with_bmus/{som_name}_with_bmus.parquet")
cfad_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/CFAD")
kde_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/KDE")
res_dir = os.path.expanduser(f"~/gpm_storm/figs/{som_name}/0_Results")
zarr_directory = "/ltenas2/data/GPM_STORM_DB/zarr"


df_bmu = pd.read_parquet(bmu_dir)
actual_combinations = set(zip(df_bmu["row"], df_bmu["col"], strict=False))
    
    
# LOAD
with open(os.path.join(res_dir,"cfad_totals.pkl"), "rb") as f:
    cfad_totals = pickle.load(f)
with open(os.path.join(res_dir,"cfad_convs.pkl"), "rb") as f:
    cfad_convs = pickle.load(f)
with open(os.path.join(res_dir,"cfad_strats.pkl"), "rb") as f:
    cfad_strats = pickle.load(f)
with open(os.path.join(res_dir,"cfad_total_tops.pkl"), "rb") as f:
    cfad_total_tops = pickle.load(f)
with open(os.path.join(res_dir,"cfad_others.pkl"), "rb") as f:
    cfad_others = pickle.load(f)

    
    
["Total", "Convective", "Stratiform"]
plot_10x10_cfad(cfad_others, cfad_dir, precip_type="Other")

node = (0,0)
individual_CFAD(cfad_total_tops, node, "Tops")
