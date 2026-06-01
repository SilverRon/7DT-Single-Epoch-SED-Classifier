# Synthetic Photometry for OSC
# %%
## Library
import os
import glob
import sys
import numpy as np
import shutil
import time
import speclite

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

### Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

### Helper Functions
import sys
sys.path.append(os.path.join('..', 'src'))
from helper import makeSpecColors
from paths import *
from var import *
from sdtpy import *

## Function
def fill_nan_in_array(arr):
    """
    입력된 numpy array에서 nan 값을 양쪽의 유효한 값들을 이용해 채우는 함수.
    
    각 nan 값에 대해:
      - 왼쪽에서 nan이 아닌 값을 찾습니다.
      - 오른쪽에서 nan이 아닌 값을 찾습니다.
      - 두 값 모두 존재하면, 두 값의 중앙값(평균)으로 대체합니다.
      - 한쪽만 존재하면, 그 값을 대체합니다.
      - 양쪽 모두 없으면 그대로 nan으로 둡니다.
    
    Parameters:
      arr (numpy.ndarray): nan이 포함된 1차원 배열
    
    Returns:
      numpy.ndarray: nan이 채워진 새로운 배열
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

# - Synthetic Photometry
def interpspecfiletrum(lam, flam, filter_lam_min=MIN_7DT_WAVELENGTH, filter_lam_max=MAX_7DT_WAVELENGTH, verbose=False):
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
	"""중복 제거 및 NaN 보정 포함한 파장/플럭스 전처리"""
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
    Process one spectrum to generate a synthetic photometry FITS table (.tmp).
    """
    # Prepare output paths
    basename = os.path.splitext(os.path.basename(specfile))[0]
    table_path = os.path.join(path_save, f"{nn:0>4}.tmp")
    plot_path = os.path.join(path_save, f"{basename}.synphot.png")
    # Skip if already done
    if os.path.exists(table_path):
        return
    # Read spectrum and preprocess flux
    # sptbl = Table.read(specfile, format='ascii')
    sptbl = Table.read(specfile,) # fits
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


# %%
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
# path_save = os.path.join(SPECTRA_WOLLAEGER_DATA, f"z{z:.3f}")
# meta_table_path = os.path.join(SPECTRA_WOLLAEGER_DATA, "z0.010/meta.csv")
# metatbl = Table.read(meta_table_path)
# logtxt += f"{len(metatbl)} rows\n"

# metatbl

# spectra = metatbl['spec'].value
spectra = sorted(glob.glob(f"{SPECTRA_DATA}/Engrave/*csv"))
types = ["KN"]*len(spectra)

path_save = os.path.join(SYNPHOT_DATA, 'Engrave', '7DT')
os.makedirs(path_save, exist_ok=True)

# %%
# Process each spectrum into a .tmp FITS table
for ss, (specfile, typ) in enumerate(zip(spectra, types)):
    print(f"[{ss+1:0>4}/{len(spectra):0>4}] Processing {os.path.basename(specfile)}", end="\r")
    process_synphot(ss, specfile, typ, path_save, sdt, fill_nan_in_array, flamunit)
    # break

tmptables = sorted(glob.glob(f"{path_save}/*.tmp"))

# %%
if len(tmptables) == len(spectra):
	print("All spectra processed successfully.")
	logtxt += "All spectra processed successfully.\n"
else:
	print(f"{len(tmptables)} out of {len(spectra)} spectra processed.")
	logtxt += f"{len(tmptables)} out of {len(spectra)} spectra processed.\n"

synphot_anomaly_class_output_table = os.path.join(path_save, "synphot_anomaly_class.csv")	

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

logtxt += f"Synphot Normal Class Output Table: {synphot_anomaly_class_output_table}\n"


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

valid_outbl.write(synphot_anomaly_class_output_table, format='csv', overwrite=True)


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

for color, typ in zip(colors, types):
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
logtxt_path = os.path.join(path_save, "log.txt")
with open(logtxt_path, 'w') as f:
    f.write(logtxt)
# %%
