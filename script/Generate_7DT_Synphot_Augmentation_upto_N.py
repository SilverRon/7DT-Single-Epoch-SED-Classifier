# --- 7DT Synthetic Photometry and Augmentation Script ---
# %%
# 1. Library Imports
import os
import glob
import sys
import numpy as np
import shutil
import time
import speclite
import uuid
import pandas as pd

start_time = time.time()

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
import warnings
warnings.filterwarnings("ignore")

# 2. Plot Settings
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# 3. Helper Functions
import sys
sys.path.append(os.path.join('..', 'src'))
from helper import makeSpecColors
from paths import *
from var import *
from sdtpy import *

# Function
def fill_nan_in_array(arr):
    """
    Fill NaN values in a 1D numpy array using nearest valid neighbor values.

    For each NaN:
      - Finds the nearest non-NaN value to the left.
      - Finds the nearest non-NaN value to the right.
      - If both exist, replaces with the median of the two values.
      - If only one exists, replaces with that value.
      - If neither exists, leaves as NaN.

    Parameters:
        arr (numpy.ndarray): 1D array potentially containing NaNs.

    Returns:
        numpy.ndarray: New array with NaNs filled.
    """
    
    # 원본 배열 복사
    filled_arr = arr.copy()
    n = len(filled_arr)
    
    # 배열의 각 원소를 순회
    for i in range(n):
        if np.isnan(filled_arr[i]):
            left_val = None
            # 왼쪽에서 nan이 아닌 값 찾기
            j = i - 1
            while j >= 0:
                if not np.isnan(filled_arr[j]):
                    left_val = filled_arr[j]
                    break
                j -= 1
            
            right_val = None

            # 오른쪽에서 nan이 아닌 값 찾기
            k = i + 1
            while k < n:
                if not np.isnan(filled_arr[k]):
                    right_val = filled_arr[k]
                    break
                k += 1

            # 대체할 값 결정
            if left_val is not None and right_val is not None:
                filled_arr[i] = np.median([left_val, right_val])
            elif left_val is not None:
                filled_arr[i] = left_val
            elif right_val is not None:
                filled_arr[i] = right_val
            # 양쪽 모두 유효한 값이 없으면 그대로 nan 유지
    return filled_arr

# 4. Synthetic Photometry Functions
def interpspecfiletrum(lam, flam, filter_lam_min=MIN_7DT_WAVELENGTH, filter_lam_max=MAX_7DT_WAVELENGTH, verbose=False):
    """
    Interpolate a spectrum to match the desired filter wavelength range by padding or trimming.

    Parameters:
        lam (Quantity): Wavelength array with units.
        flam (Quantity): Flux array with units.
        filter_lam_min (float): Minimum filter wavelength.
        filter_lam_max (float): Maximum filter wavelength.
        verbose (bool): If True, print warning messages.

    Returns:
        tuple: (interp_lam, interp_flam, flagtype) where:
            interp_lam (ndarray): Padded wavelength array.
            interp_flam (ndarray): Corresponding flux array with NaNs filled.
            flagtype (str): Indicates padding type ('clean', 'blue', 'red', or 'blue_red').
    """
    #	Default Flag Type
    flagtype = 'TBD'

    #	Median Resolution
    median_res = np.median(lam.value[1:] - lam.value[:-1])

    #	Wavelength Range
    lammin, lammax = lam.value.min(), lam.value.max()

    #	Padding
    left_pad = np.mean(flam.value[:10])
    right_pad = np.mean(flam.value[-10:])

    #	Interpolation
    ##	Spectra has a shorter wavelength range than the filters
    if (lammin > filter_lam_min) & (lammax < filter_lam_max):
        if verbose:
            print(f"Warning! {lammin} < {filter_lam_min} and {lammax} > {filter_lam_max}")

        bluelamarr = np.arange(filter_lam_min-median_res, lammin, median_res)
        redlamarr = np.arange(lammax, filter_lam_max+median_res, median_res)

        interp_lam = np.concatenate((bluelamarr, lam.value, redlamarr))
        interp_flam = np.concatenate((
            np.full(len(bluelamarr), left_pad),
            flam.value,
            np.full(len(redlamarr), right_pad)
        ))
        flagtype = 'blue_red'

    ##	Spectra has a longer wavelength range than the filters
    elif (lammin > filter_lam_min):
        if verbose:
            print(f"Warning! {lammin} > {filter_lam_min}")
        bluelamarr = np.arange(filter_lam_min-median_res, lammin, median_res)
        # blueflamarr = np.interp(bluelamarr, lam.value, flam.value)
        interp_lam = np.concatenate((bluelamarr, lam.value))
        interp_flam = np.concatenate((np.full(len(bluelamarr), left_pad), flam.value))
        flagtype = 'blue'

    ##	Spectra has a longer wavelength range than the filters
    elif (lammax < filter_lam_max):
        if verbose:
            print(f"Warning! {lammax} > {filter_lam_max}")
        redlamarr = np.arange(lammax, filter_lam_max+median_res, median_res)
        interp_lam = np.concatenate((lam.value, redlamarr))
        interp_flam = np.concatenate((flam.value, np.full(len(redlamarr), right_pad)))
        flagtype = 'red'

    ##	Spectra is already in the range of filters
    else:
        if verbose:
            print(f"Spectrum is already in the range of filters.")
        interp_lam = lam.value
        interp_flam = flam.value

        flagtype = 'clean'

    return (interp_lam, fill_nan_in_array(interp_flam), flagtype)



### Functions for synthetic photometries 

def process_flux(lam, flam, unit_lam=lamunit, unit_flux=flamunit):
    """
    Preprocess wavelength and flux arrays by sorting, removing duplicates, and filling NaNs.

    Parameters:
        lam (array-like): Wavelength values.
        flam (array-like): Flux values.
        unit_lam (astropy.units.Unit): Unit for wavelengths.
        unit_flux (astropy.units.Unit): Unit for flux values.

    Returns:
        tuple: (lam * unit_lam, flam * unit_flux)
    """
    lam = np.array(lam)
    flam = np.array(flam)

    # 정렬
    sort_idx = np.argsort(lam)
    lam = lam[sort_idx]
    flam = flam[sort_idx]

    # 중복 제거
    _, unique_idx = np.unique(lam, return_index=True)
    lam = lam[unique_idx]
    flam = flam[unique_idx]

    # NaN 보정
    if np.any(np.isnan(flam)):
        print("  - NaN in flux → interpolated")
        flam = fill_nan_in_array(flam)

    return lam * unit_lam, flam * unit_flux


# New function for processing one spectrum to a synthetic photometry FITS table (.tmp)
def process_synphot(nn, specfile, typ, path_save, sdt, fill_nan_in_array, flamunit):
    """
    Process a single spectrum file to produce synthetic photometry and save results.

    Parameters:
        nn (int): Index for output filename formatting.
        specfile (str): Path to the ASCII spectrum file.
        typ (str): Transient type label.
        path_save (str): Directory to save output .tmp and plot files.
        sdt (SevenDT): Instance for synthetic photometry calculations.
        fill_nan_in_array (callable): Function to fill NaNs in flux arrays.
        flamunit (astropy.units.Unit): Unit of input flux values.

    Returns:
        None
    """
    # Prepare output paths
    basename = os.path.splitext(os.path.basename(specfile))[0]
    table_path = os.path.join(path_save, f"{nn:0>4}.tmp")
    plot_path = os.path.join(path_save, f"{basename}.synphot.png")
    # Skip if already done
    if os.path.exists(table_path):
        return
    # Read spectrum and preprocess flux
    sptbl = Table.read(specfile, format='ascii')
    keys = sptbl.keys()
    lam, flam = process_flux(sptbl[keys[0]], sptbl[keys[1]])
    # Generate synphot observations
    mobstbl = sdt.get_synphot2obs(
        flam, lam, z=None, z0=None,
        figure=(not os.path.exists(plot_path))
    )
    # Save plot if generated
    if not os.path.exists(plot_path):
        plt.title(basename)
        plt.savefig(plot_path)
        plt.close()
    # Build and write output FITS table
    outbl = Table()
    outbl['type'] = [typ]
    outbl['spec'] = [specfile.strip()]
    outbl['flagtype'] = [ 'clean' ]
    for filte in mobstbl['filter']:
        idx = np.where(mobstbl['filter'] == filte)
        for key in mobstbl.keys()[3:]:
            colname = f"{key}_{filte}"
            outbl[colname] = mobstbl[key][idx].item()
            outbl[colname].format = '1.3f'
    outbl.write(table_path, format='fits')

# For Augmentatio
def process_synphot_augmentation(nn, specfile, typ, path_save, sdt, number_of_iterations=1, flux_scaleing_factors=None):   
    """
    Process a single spectrum file to produce synthetic photometry and save results.

    Parameters:
        nn (int): Index for output filename formatting.
        specfile (str): Path to the ASCII spectrum file.
        typ (str): Transient type label.
        path_save (str): Directory to save output .tmp and plot files.
        sdt (SevenDT): Instance for synthetic photometry calculations.
        fill_nan_in_array (callable): Function to fill NaNs in flux arrays.
        flamunit (astropy.units.Unit): Unit of input flux values.

    Returns:
        None
    """
    # Prepare output paths
    basename = os.path.splitext(os.path.basename(specfile))[0]
    table_path = os.path.join(path_save, f"{nn:0>4}.tmp")
    # plot_path = os.path.join(path_save, f"{basename}.synphot.png")
    # Skip if already done
    if os.path.exists(table_path):
        return
    # Read spectrum and preprocess flux
    sptbl = Table.read(specfile, format='ascii')
    keys = sptbl.keys()
    lam, flam = process_flux(sptbl[keys[0]], sptbl[keys[1]])

    n_augmented = len(flux_scaleing_factors) * number_of_iterations

    tables = []
    # Augmentation
    for n_iter in range(number_of_iterations):
        for scale_factor in flux_scaleing_factors:
            flam_scaled = flam * scale_factor
            # Generate synphot observations
            sub_mobstbl = sdt.get_synphot2obs(
                flam_scaled, lam, z=None, z0=None, figure=False)
            
            sub_mobstbl['n_iter'] = n_iter
            sub_mobstbl['scale_factor'] = scale_factor
            tables.append(sub_mobstbl)

    # Build and write output FITS table
    outbl = Table()
    outbl['type'] = [typ]*n_augmented
    outbl['spec'] = [specfile.strip()]*n_augmented
    outbl['flagtype'] = ['clean_augmented']*n_augmented

    #   Make empty columns
    for filte in sub_mobstbl['filter']:
        for key in sub_mobstbl.keys()[3:]:
            colname = f"{key}_{filte}"
            outbl[colname] = -99.

    outbl['n_iter'] = int(99)
    outbl['scale_factor'] = float(-99)

    #   Convert sub_mobstbl to outbl
    for nn, sub_mobstbl in enumerate(tables):
        for filte in sub_mobstbl['filter']:
            idx = np.where(sub_mobstbl['filter'] == filte)
            for key in sub_mobstbl.keys()[3:]:
                colname = f"{key}_{filte}"
                outbl[colname][nn] = sub_mobstbl[key][idx].item()
                # outbl[colname][nn].format = '1.3f'
        outbl['n_iter'][nn] = sub_mobstbl['n_iter'][idx].item()
        outbl['scale_factor'][nn] = sub_mobstbl['scale_factor'][idx].item()
    #   Write outbl to table_path
    outbl.write(table_path, format='fits')


# %%
CLASS_CONFIG = os.path.join(CONFIG, 'class_config.yaml')
with open(CLASS_CONFIG, 'r') as f:
    class_config = yaml.safe_load(f)
normal_class = class_config['CLASSES_TO_CLASSIFY']['NORMAL_CLASS']
anomalous_class = class_config['CLASSES_TO_CLASSIFY']['ANOMALOUS_CLASS']
validation_class = class_config['CLASSES_TO_CLASSIFY']['VALIDATION_CLASS']

class_config
# %%
# 5. Main Execution
# 7DT Setting
sys.path.append('..')
from simulator.helper import *
from simulator.sdtpy import *
register_custom_filters_on_speclite('../simulator')

#	Exposure Time [s]
sdt = SevenDT()
sdt.echo_optics()
filterset = sdt.generate_filterset(bandmin=BANDMIN, bandmax=BANDMAX, bandwidth=BANDWIDTH, bandstep=BANDSTEP, bandrsp=BANDRSP, lammin=LAMMIN, lammax=LAMMAX, lamres=LAMRES)
T_qe = sdt.get_CMOS_IMX455_QE()
sdt.get_optics()
s = sdt.get_sky()
sdt.smooth_sky()
totrsptbl = sdt.calculate_response()
Npix_ptsrc, Narcsec_ptsrc = sdt.get_phot_aperture(exptime=EXPTIME_SINGLE, fwhm_seeing=SEEING, optfactor=EFF_FACTOR, verbose=False)
depthtbl = sdt.get_depth_table(Nsigma=5)
sdt.get_speclite()

# %%
depth_dict = {}
for filtername, depth in zip(depthtbl['name'], depthtbl['5sigma_depth']):
	depth_dict[filtername] = depth

depth_dict
# %%
# Data
logtxt = ""
## Meta Table containing spectra paths
meta_table_path = os.path.join(SPECTRA_DATA, "merged_osc_wiserep_spectra_meta_table.csv")
# metatbl = Table.read(meta_table_path)
metatbl = pd.read_csv(meta_table_path)
logtxt += f"{len(metatbl)} rows\n"
# %%
normal_metatbl = Table.from_pandas(metatbl[metatbl['broad_type'].isin(normal_class)])
anomalous_metatbl = Table.from_pandas(metatbl[metatbl['broad_type'].isin(anomalous_class)])
validation_metatbl = Table.from_pandas(metatbl[metatbl['broad_type'].isin(validation_class)])

print(f"Normal Class: {len(normal_metatbl)} rows")
print(f"Anomalous Class: {len(anomalous_metatbl)} rows")
print(f"Validation Class: {len(validation_metatbl)} rows")

logtxt += f"Normal Class: {len(normal_metatbl)} rows\n"
normal_types, normal_type_counts = np.unique(normal_metatbl['broad_type'], return_counts=True)
logtxt += "Normal Class Types: {}\n".format(normal_types.value)
logtxt += "Normal Class Type Counts: {}\n".format(normal_type_counts)
print("Normal Class Types: {}".format(normal_types.value))
print("Normal Class Type Counts: {}".format(normal_type_counts))

logtxt += f"Anomalous Class: {len(anomalous_metatbl)} rows\n"
anomalous_types, anomalous_type_counts = np.unique(anomalous_metatbl['broad_type'], return_counts=True)
logtxt += "Anomalous Class Types: {}\n".format(anomalous_types.value)
logtxt += "Anomalous Class Type Counts: {}\n".format(anomalous_type_counts)
print("Anomalous Class Types: {}".format(anomalous_types.value))
print("Anomalous Class Type Counts: {}".format(anomalous_type_counts))

logtxt += f"Validation Class: {len(validation_metatbl)} rows\n"
validation_types, validation_type_counts = np.unique(validation_metatbl['broad_type'], return_counts=True)
logtxt += "Validation Class Types: {}\n".format(validation_types.value)
logtxt += "Validation Class Type Counts: {}\n".format(validation_type_counts)
print("Validation Class Types: {}".format(validation_types.value))
print("Validation Class Type Counts: {}".format(validation_type_counts))
# %%
# Only Normal class will be augmented
spectra = normal_metatbl['path_spectra_file'].value
types = normal_metatbl['broad_type'].value
# %%
#
# Augmentation Setting
# Log-uniform sampling for flux scaling factors will be used per spectrum to reach n_target_per_class per class.

# Function for log-uniform sampling of flux scaling factors
def sample_flux_scaling_factors(n, low=0.1, high=10.0, seed=None):
    rng = np.random.default_rng(seed)
    return 10 ** rng.uniform(np.log10(low), np.log10(high), size=n)

#
# Desired target per class
n_target_per_class = 30000  # e.g., 10k per class
unique_types, type_counts = np.unique(types, return_counts=True)

logtxt += f"Set n_target_per_class to {n_target_per_class}\n"
logtxt += "# Per-class augmentation: Each class will reach n_target_per_class using log-uniform flux scaling factors.\n"
for t, count in zip(unique_types, type_counts):
    # Use ceil division to guarantee at least n_target_per_class samples per class
    n_target = int(np.ceil(n_target_per_class / count))
    logtxt += f"  {t}: {n_target} augmentations per spectrum (total spectra: {count})\n"
logtxt += "\n"

print("Per-class augmentations (to reach n_target_per_class):")
for t, count in zip(unique_types, type_counts):
    # Use ceil division to guarantee at least n_target_per_class samples per class
    n_target = int(np.ceil(n_target_per_class / count))
    print(f"  {t}: {n_target} augmentations per spectrum (total spectra: {count})")

# %%
#
# For each spectrum, create per-class augmentation so that each class reaches n_target_per_class
# The number of augmentations per spectrum is now set by log-uniform sampling of flux scaling factors.
for ss, (specfile, typ) in enumerate(zip(spectra, types)):
    print(f"[{ss+1:0>4}/{len(spectra):0>4}] Processing {os.path.basename(specfile)}", end="\r")
    # Use ceil division to guarantee at least n_target_per_class samples per class
    n_target = int(np.ceil(n_target_per_class / type_counts[np.where(unique_types == typ)[0][0]]))
    flux_scaleing_factors_this = sample_flux_scaling_factors(n_target, low=0.1, high=10.0, seed=ss+42)
    flux_scaleing_factors_this = np.append(flux_scaleing_factors_this, 1.0) # original one
    process_synphot_augmentation(ss, specfile, typ, AUGMENTED_BALANCED_PHOT_7DT_DATA, sdt, number_of_iterations=1, flux_scaleing_factors=flux_scaleing_factors_this)

tmptables = sorted(glob.glob(f"{AUGMENTED_BALANCED_PHOT_7DT_DATA}/*.tmp"))

# %%
if len(tmptables) == len(spectra):
    print("All spectra processed successfully with per-class log-uniform flux scaling augmentation.")
    logtxt += "All spectra processed successfully with per-class log-uniform flux scaling augmentation.\n"
else:
    print(f"{len(tmptables)} out of {len(spectra)} spectra processed.")
    logtxt += f"{len(tmptables)} out of {len(spectra)} spectra processed.\n"

synphot_normal_class_output_table = os.path.join(AUGMENTED_BALANCED_PHOT_7DT_DATA, "synphot_normal_class.csv")	

# %%
# Read and stack FITS tables, converting byte-strings to unicode
outbl = vstack([
    Table.read(_outname, format='fits', memmap=False)
    for _outname in tmptables
])
# Decode byte-string spec paths to Unicode and strip padding
columns_to_decode = ['type', 'spec', 'flagtype']

for col in columns_to_decode:
    outbl[col] = [
        spec_path.decode('utf-8').strip() if isinstance(spec_path, (bytes, bytearray)) else str(spec_path).strip()
        for spec_path in outbl[col]
    ]

logtxt += f"Synphot Normal Class Output Table: {synphot_normal_class_output_table}\n"


# %%
previous_filter_names = [col.split('_')[1] for col in outbl.columns if col.startswith('magabs_')]
new_filter_names = [col[:-1] for col in previous_filter_names]


filter_name_map_dict = {}
for prev_name, new_name in zip(previous_filter_names, new_filter_names):
	filter_name_map_dict[prev_name] = new_name

for col in outbl.columns:
    if any([col.startswith(prev_name) for prev_name in previous_filter_names]):
        for prev_name, new_name in filter_name_map_dict.items():
            if col.startswith(prev_name):
                outbl.rename_column(col, col.replace(prev_name, new_name))

# %%
# %%
#	Convert the table format
for key in outbl.keys():
	for filte in previous_filter_names:
		if filte in key:
			# print(key.replace(filte, new_filter_names[previous_filter_names.index(filte)]))
			outbl[key].name = key.replace(filte, new_filter_names[previous_filter_names.index(filte)])
		

# %%
# Dynamically filter out rows with NaN in any magobs filter
filter_cols = [f'magobs_{band}' for band in new_filter_names]
# Build a boolean mask where all specified columns are non-NaN
indx_valid = np.logical_and.reduce([~np.isnan(outbl[col]) for col in filter_cols])
valid_outbl = outbl[indx_valid]

print(f"Number of valid spectra: {len(valid_outbl)} among {len(outbl)} ({len(valid_outbl)/len(outbl):.1%})")
logtxt += f"Number of valid spectra: {len(valid_outbl)} among {len(outbl)} ({len(valid_outbl)/len(outbl):.1%})\n"

valid_outbl.write(synphot_normal_class_output_table, format='csv', overwrite=True)


# %%
# flags, flag_counts = np.unique(outbl['flagtype'], return_counts=True)
# plt.bar(flags, flag_counts, alpha=0.7)
# plt.yscale('log')
# plt.ylabel("Number of Spectra")
# plt.xlabel("Flag Type")
# plt.show()
# %%
# Color–color scatter per transient type with distinct colors
fig = plt.figure(figsize=(10, 10))
types = np.unique(outbl['type'])
cmap = plt.get_cmap('tab20')  # qualitative colormap
colors = cmap(np.linspace(0, 1, len(types)))

filter0 = 'g'
filter1 = 'r'
filter2 = 'i'

rubin_7DT_map_dict = {'u': 'm400', 'g': 'm475', 'r': 'm625', 'i': 'm750', 'z': 'm875', 'y': 'm887'}

for color, typ in zip(colors, types[::-1]):
    typtbl = outbl[outbl['type'] == typ]
    gr_colors = typtbl[f'magabs_{rubin_7DT_map_dict[filter0]}'] - typtbl[f'magabs_{rubin_7DT_map_dict[filter1]}']
    ri_colors = typtbl[f'magabs_{rubin_7DT_map_dict[filter1]}'] - typtbl[f'magabs_{rubin_7DT_map_dict[filter2]}']
    plt.scatter(gr_colors, ri_colors, c=[color], alpha=0.5, label=typ, edgecolors='none')

plt.xlabel(f"{rubin_7DT_map_dict[filter0]}-{rubin_7DT_map_dict[filter1]}")
plt.ylabel(f"{rubin_7DT_map_dict[filter1]}-{rubin_7DT_map_dict[filter2]}")
plt.xlim(-0.5, 1.5)
plt.ylim(-1, 0.75)
plt.legend()
plt.grid('both', ls='--', alpha=0.5)
plt.tight_layout()
plotname = os.path.join(AUGMENTED_BALANCED_PHOT_7DT_DATA, "color-color.png")
plt.savefig(plotname)
plt.show()

# %%
problematic_specfiles = outbl['spec'][~indx_valid].value
print(f"Number of problematic spectra: {len(problematic_specfiles)}")
logtxt += f"Number of problematic spectra: {len(problematic_specfiles)}\n"
problematic_specfiles

# %%
delt = time.time() - start_time
print(f"Time taken: {delt:.2f} seconds")
logtxt += f"Time taken: {delt:.2f} seconds\n"

logtxt += "END\n"
logtxt_path = os.path.join(AUGMENTED_BALANCED_PHOT_7DT_DATA, "log.txt")
with open(logtxt_path, 'w') as f:
    f.write(logtxt)