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


def extrapolate_with_trend(lam, flam, filter_lam_min, filter_lam_max,
                           edge_width=10, median_base=5, verbose=False):
    """
    스펙트럼의 앞/뒤로 파장대가 부족할 경우, 선형 추세에 기반한 외삽을 수행한다.
    """
    lam = lam.value if hasattr(lam, 'value') else lam
    flam = flam.value if hasattr(flam, 'value') else flam

    median_res = np.median(np.diff(lam))
    lammin, lammax = lam.min(), lam.max()

    interp_lam = lam
    interp_flam = flam
    flagtype = 'clean'

    if lammin > filter_lam_min:
        n_steps = int((lammin - filter_lam_min) // median_res)
        bluelamarr = np.linspace(filter_lam_min, lammin - median_res, n_steps)

        # 변화량 계산: 최근 edge_width개에서 차이 → 중앙값
        df = np.diff(flam[:edge_width])
        df_median = np.median(df)

        # 시작값: edge 시작부 5~10개 flux의 중앙값
        f_start = np.median(flam[:median_base])
        blueflamarr = f_start + df_median * np.arange(-n_steps, 0)

        interp_lam = np.concatenate((bluelamarr, interp_lam))
        interp_flam = np.concatenate((blueflamarr, interp_flam))
        flagtype = 'blue'

    if lammax < filter_lam_max:
        n_steps = int((filter_lam_max - lammax) // median_res)
        redlamarr = np.linspace(lammax + median_res, filter_lam_max, n_steps)

        df = np.diff(flam[-edge_width:])
        df_median = np.median(df)

        f_start = np.median(flam[-median_base:])
        redflamarr = f_start + df_median * np.arange(1, n_steps + 1)

        interp_lam = np.concatenate((interp_lam, redlamarr))
        interp_flam = np.concatenate((interp_flam, redflamarr))
        flagtype = 'red' if flagtype == 'clean' else 'blue_red'

    if verbose:
        print(f"[{flagtype}] Extrapolated with median trend | "
              f"Δf_blue = {df_median:.3e}, Δf_red = {df_median:.3e} (if applied)")

    return interp_lam, interp_flam, flagtype

def plotspecfiletrum_abmag(interp_lam, interp_flam, filter_lam_min=MIN_7DT_WAVELENGTH, filter_lam_max=MAX_7DT_WAVELENGTH, flagtype=None):
	mags = [mag for mag in lsst.get_ab_magnitudes(interp_flam*flamunit, interp_lam*lamunit).as_array()[0]]

	plt.plot(interp_lam, convert_flam2fnu(interp_flam*flamunit, interp_lam*lamunit).to(u.ABmag), c='k', alpha=0.5)
	plt.plot(eff_wavelengths, mags, 'o-', c='k', alpha=1.0)
	for jj, (filte, color) in enumerate(colors.items()):
		if ((flagtype == 'blue_red') & ((jj == 0) | (jj == len(colors)-1))) | ((flagtype == 'red') & (jj == len(colors)-1)) | ((flagtype == 'blue') & (jj == 0)):
			marker = 'D'
			label = f"{filte} (extrapolated)"
		else:
			marker = 's'
			label = f"{filte}"
		plt.plot(eff_wavelengths[jj], mags[jj], f'{marker}-', mec=color, mfc='w', mew=3, ms=10, c=color, label=label)

	yl, yu = plt.ylim()
	yu = np.nanmax(mags) + 1.0
	yl -= 1.0
	plt.ylim(yu, yl)
	plt.xlim(filter_lam_min, filter_lam_max)
	plt.ylabel("AB mag")
	plt.xlabel("Wavelength (Angstrom)")
	plt.legend(loc='upper left', frameon=False)
	plt.tight_layout()
	
def calculate_snr(mag, depth_5sigma):
	deltamag = mag-depth_5sigma
	return 5*10**(-0.4*(deltamag))

def get_random_point(mu, sigma, n=10):
	"""
	mu, sigma = 17.5, 0.1
	n = 10
	"""
	x = np.arange(mu-sigma*n, mu+sigma*n, sigma*1e-3)
	y = norm(mu, sigma).pdf(x)
	return np.random.choice(x, p=y/np.sum(y))

def calculate_magobs(mag, snr, zperr=0.0, n=10):

	#	Signal-to-noise ratio (SNR)
	
	#	SNR --> mag error
	merr0 = convert_snr2magerr(snr)
	#	Random obs points
	m = get_random_point(mag, merr0, n=n)
	#	Measured error
	merr = np.sqrt(merr0**2+zperr**2)

	return (m, merr)

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


def process_spectrum(nn, specfile, typ, path_save, draw_plot=True):
	"""
	Process a spectrum file and save the output table.
	
	Parameters
	----------
	nn : int
		spectrum index
	specfile : str
		spectrum file path
	typ : str
		spectrum type
	path_save : str
		path to save the output table
	"""	
	#	File Name
	basename = os.path.splitext(os.path.basename(specfile))[0]
	table_path = os.path.join(path_save, f"{nn:0>4}.tmp")
	plot_path = os.path.join(path_save, f"{basename}.png")

	if os.path.exists(table_path) or os.path.exists(plot_path):
		return

	# Read Spectrum
	# sptbl = Table.read(specfile, format='ascii')
	sptbl = Table.read(specfile,)
	keys = sptbl.keys()
	lam, flam = process_flux(sptbl[keys[0]], sptbl[keys[1]])

	# Interpolation
	interp_lam, interp_flam, flagtype = interpspecfiletrum(
		lam, flam,
		filter_lam_min=filter_lam_min,
		filter_lam_max=filter_lam_max
	)

	# Remove Duplicates
	interp_lam, interp_flam = map(np.array, (interp_lam, interp_flam))
	_, unique_idx = np.unique(interp_lam, return_index=True)
	interp_lam = interp_lam[unique_idx]
	interp_flam = interp_flam[unique_idx]

	# AB Magnitude Calculation
	mags = lsst.get_ab_magnitudes(interp_flam * flamunit, interp_lam * lamunit).as_array()[0]

	# Plot Saving
	if draw_plot:
		plotspecfiletrum_abmag(interp_lam, interp_flam,
								filter_lam_min=filter_lam_min, filter_lam_max=filter_lam_max,
								flagtype=flagtype)
		plt.title(f"{typ})")
		plt.tight_layout()
		plt.savefig(plot_path, dpi=100)
		plt.close()

	# Result Table Saving
	outbl = Table()
	outbl['type'] = [typ]
	outbl['spec'] = [specfile.strip()]
	outbl['flagtype'] = [flagtype]

	for ff, filte in enumerate(filters):
		outbl[f"magabs_{filte}"] = [mags[ff]]
		outbl[f"magabs_{filte}"].format = '1.6f'

	outbl.write(table_path, format='fits')

# ✅ 메인 루프
# for nn, specfile in enumerate(spectra):
#     print(f"[{nn+1}/{len(spectra)} ({(nn+1)/len(spectra):.1%})] {specfile}" + " " * 50, end="\r")
#     process_spectrum(nn, specfile, rep_wmetatbl, path_save)


# %%
## Rubin/LSST Setting
lsst = speclite.filters.load_filters('lsst2023-*')
filter_lam_min, filter_lam_max = 3000, 11000
speclite.filters.plot_filters(
    lsst, wavelength_limits=(filter_lam_min, filter_lam_max), legend_loc='upper left')

eff_wavelengths = lsst.effective_wavelengths.value
filters = [val.split('-')[-1] for val in lsst.names]
filters
colors = {
    'u': '#a05eb5',
    'g': '#1f77b4',
    'r': '#2ca02c',
    'i': '#98df8a',
    'z': '#ffbb78',
    'y': '#d62728',
}
depth = {
	"u": 23.9,
	"g": 25.0,
	"r": 24.7,
	"i": 24.0,
	"z": 23.3,
	"y": 22.1,
}

# %%
# Data
logtxt = ""
## Meta Table containing spectra paths
# meta_table_path = os.path.join(SPECTRA_WOLLAEGER_DATA, "z0.010/meta.csv")
# metatbl = Table.read(meta_table_path)
# logtxt += f"{len(metatbl)} rows\n"

# metatbl


# spectra = metatbl['spec'].value
spectra = sorted(glob.glob(f"{SPECTRA_DATA}/Engrave/*csv"))
types = ["KN"]*len(spectra)

path_save = os.path.join(SYNPHOT_DATA, 'Engrave', 'Rubin')
os.makedirs(path_save, exist_ok=True)# %%

# specfile = spectra[0]
# sptbl = Table.read(specfile, format='ascii')
# keys = sptbl.keys()
# fig = plt.figure(figsize=(15, 5))
# plt.subplot(131)
# plt.plot(lam, flam)
# plt.subplot(132)
# lam, flam = process_flux(sptbl[keys[0]], sptbl[keys[1]])
# plt.subplot(133)
# interp_lam, interp_flam, flagtype = interpspecfiletrum(
# 	lam, flam,
# 	filter_lam_min=filter_lam_min,
# 	filter_lam_max=filter_lam_max
# )
# plt.plot(interp_lam, interp_flam)

# interp_lam2, interp_flam2, flagtype = extrapolate_with_trend(lam, flam, filter_lam_min, filter_lam_max,
# 						   edge_width=10, median_base=5, verbose=False)
# plt.plot(interp_lam2, interp_flam2)
# plt.show()


# %%
## Iteration
#	Repeated Table
faildict = {}

###	Parallel Processing
for nn, (specfile, typ) in enumerate(zip(spectra, types)):
	print(f"[{nn+1:0>4}/{len(spectra):0>4} ({(nn+1)/len(spectra):.1%})] {specfile}" + " " * 50, end="\r")
	process_spectrum(nn, specfile, typ, path_save)

tmptables = sorted(glob.glob(f"{path_save}/*.tmp"))

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


#	Convert the table format
for filte in filters:
	# Magnitude (Truth)
	magabsarr = outbl[f'magabs_{filte}']
	# Depth
	depth_5sigma = depth[filte]
	# SNR
	outbl[f"snr_{filte}"] = calculate_snr(magabsarr, depth_5sigma=depth_5sigma)
	snrarr = calculate_snr(magabsarr, depth_5sigma)
	magobs = np.zeros(len(magabsarr))
	magerr = np.zeros(len(magabsarr))
	for nn, (magabs, snrval) in enumerate(zip(magabsarr, snrarr)):
		try:
			_magobs, _magerr = calculate_magobs(magabs, snrval, zperr=0.0, n=10)
		except ValueError as e:
			_magobs, _magerr = np.nan, np.nan
			print(f"{nn}: {magabs}, {snrval}")
			print(f"Error in calculate_magobs: {e}")
		magobs[nn] = _magobs
		magerr[nn] = _magerr
	#
	outbl[f"magobs_{filte}"] = magobs
	outbl[f"magerr_{filte}"] = magerr
	outbl[f"snr_{filte}"] = snrarr
print(f"DONE!")

# %%
indx_valid = (~np.isnan(outbl['magobs_u'])) & (~np.isnan(outbl['magobs_g'])) & (~np.isnan(outbl['magobs_r'])) & (~np.isnan(outbl['magobs_i'])) & (~np.isnan(outbl['magobs_z'])) & (~np.isnan(outbl['magobs_y']))
valid_outbl = outbl[indx_valid]

print(f"Number of valid spectra: {len(valid_outbl)} among {len(outbl)} ({len(valid_outbl)/len(outbl):.1%})")
logtxt += f"Number of valid spectra: {len(valid_outbl)} among {len(outbl)} ({len(valid_outbl)/len(outbl):.1%})\n"

valid_outbl.write(synphot_anomaly_class_output_table, format='csv', overwrite=True)


# %%
flags, flag_counts = np.unique(outbl['flagtype'], return_counts=True)
# print(flags, flag_counts)
# logtxt += f"Flags: {flags}\n"
# logtxt += f"Flag Counts: {flag_counts}\n"
plt.bar(flags, flag_counts, alpha=0.7)
plt.yscale('log')
plt.ylabel("Number of Spectra")
plt.xlabel("Flag Type"
)
plt.show()
# %%
# Color–color scatter per transient type with distinct colors
fig = plt.figure(figsize=(10, 10))
types = np.unique(outbl['type'])
cmap = plt.get_cmap('tab20')  # qualitative colormap
colors = cmap(np.linspace(0, 1, len(types)))

for color, typ in zip(colors, types):
    typtbl = outbl[outbl['type'] == typ]
    gr_colors = typtbl['magabs_g'] - typtbl['magabs_r']
    ri_colors = typtbl['magabs_r'] - typtbl['magabs_i']
    plt.scatter(gr_colors, ri_colors, c=[color], alpha=0.5, label=typ, edgecolors='none')

plt.xlabel("g-r")
plt.ylabel("r-i")
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
logtxt_path = os.path.join(path_save, 'log.txt')
with open(logtxt_path, 'w') as f:
    f.write(logtxt)