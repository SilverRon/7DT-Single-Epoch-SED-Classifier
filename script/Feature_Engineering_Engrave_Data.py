# Setting
## Library
import os
import glob
import sys
import json
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
import warnings
warnings.filterwarnings("ignore")
# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Jupyter Setting
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
def add_semi_global_color(filter_columns, num_filters, interval):
    filters = []
    for i in range(0, len(filter_columns) - num_filters + 1, interval):
        subset = filter_columns[i:i + num_filters]
        semi_global_name = f"c{num_filters:g}_{subset[0].split('_')[1][0:4]}_{subset[-1].split('_')[1][1:4]}"
        semi_global_value = sum(photometry_table[filt] for filt in subset) / num_filters
        filters.append((semi_global_name, semi_global_value))
    return filters
# Data
logtxt = ""
photometry_table_file = os.path.join(PHOT_NEW_DATA, "final_synphot_normal_class.csv")
photometry_table = Table.read(photometry_table_file)
print(f"{len(photometry_table):,} photometry table read")
logtxt += f"{len(photometry_table):,} photometry table read\n"
## UID Generation

LSST_FILTER_EFF_WAVELENGTHS = {key: val/10. for key, val in LSST_FILTER_EFF_WAVELENGTHS.items()}

# UID Generation
import uuid

uid_length = 8
# photometry_table은 'spec' 컬럼을 가진 astropy Table이라고 가정합니다.
# 재현성을 위해 고정된 네임스페이스를 사용합니다.
namespace = uuid.NAMESPACE_URL

# 각 unique spec 추출
unique_specs = np.unique(photometry_table['spec'])
print(f"{len(unique_specs):,} unique spectra")
logtxt += f"{len(unique_specs):,} unique spectra\n"

# 각 unique spec에 대해 고유한 UID 생성 (uuid5를 사용하면 같은 spec은 항상 같은 UID가 생성됨)
uid_mapping = {spec: str(uuid.uuid5(namespace, spec))[:uid_length] for spec in unique_specs}
print(f"{len(uid_mapping):,} unique UID generated")
logtxt += f"{len(uid_mapping):,} unique UID generated\n"

# photometry_table에 'uid'라는 새로운 컬럼 추가: 각 행의 spec에 대응하는 UID 할당
photometry_table['uid'] = [uid_mapping[spec] for spec in photometry_table['spec']]
print(f"{len(photometry_table):,} photometry table with uid")
logtxt += f"{len(photometry_table):,} photometry table with uid\n"
# %%
# Feature Engineering
logtxt += "\nFeature Engineering\n"
## 40 Features
logtxt += "\n40 Features\n"
feature40_dict = {}
import itertools

# 40개 필터의 이름 추출
filters = MEDIUM_BANDS
filter_columns = [f'magobs_{filter}' for filter in filters]

# 새로운 피처 테이블 생성
eng_table = Table()
eng_table['Sample_ID'] = np.arange(len(photometry_table))
eng_table['Class'] = photometry_table['type']  # 클래스 열이 있다고 가정
basic_keys = ["Sample_ID", "Class", "uid"]
# Feature Engineering
## 40 Filters
### 1. Color
keys = []
color_combinations = list(itertools.combinations(filter_columns, 2))
for f1, f2 in color_combinations:
    color_name = f"{f1.split('_')[1][:4]}-{f2.split('_')[1][:4]}"
    eng_table[color_name] = photometry_table[f1] - photometry_table[f2]
    keys.append(color_name)
feature40_dict['color'] = keys

n_color_features = len(keys)
logtxt += f"{n_color_features} color features generated\n"
### 2. Residual
# - res1
# - res2
keys = []
for i, filter_col in enumerate(filter_columns):
    if i > 0 and i < len(filter_columns) - 1:
        left = photometry_table[filter_columns[i - 1]]
        right = photometry_table[filter_columns[i + 1]]
        continuum = (left + right) / 2
        residual_name = f"r1_{filter_col.split('_')[1][:4]}"
        eng_table[residual_name] = photometry_table[filter_col] - continuum
        keys.append(residual_name)
feature40_dict['res1'] = keys

n_res1_features = len(keys)
logtxt += f"{n_res1_features} residual features generated\n"
keys = []
for i, filter_col in enumerate(filter_columns):
    if i > 1 and i < len(filter_columns) - 2:
        left = (photometry_table[filter_columns[i - 2]] + photometry_table[filter_columns[i - 1]]) / 2
        right = (photometry_table[filter_columns[i + 1]] + photometry_table[filter_columns[i + 2]]) / 2
        continuum = (left + right) / 2
        double_residual_name = f"r2_{filter_col.split('_')[1][:4]}"
        eng_table[double_residual_name] = photometry_table[filter_col] - continuum
        keys.append(double_residual_name)
feature40_dict['res2'] = keys

n_res2_features = len(keys)
logtxt += f"{n_res2_features} double residual features generated\n"

### Pseudo Color
# - 2, 4, 8, 10, 20
# 모든 필터 묶기와 컬러 생성하기
semi_global_sets = {}
numbers_to_bind = [2, 4, 8, 10, 20]
for number_to_bind in numbers_to_bind:
	semi_global_sets[number_to_bind] = add_semi_global_color(filter_columns, number_to_bind, int(number_to_bind / 2))
	#
	keys = []
	for num_filters, columns in semi_global_sets.items():
		color_combinations = itertools.combinations(columns, 2)  # 2개씩 조합
		for (c1_name, c1_value), (c2_name, c2_value) in color_combinations:
			color_name = f"{c1_name}-{c2_name}"
			eng_table[color_name] = c1_value - c2_value
			keys.append(color_name)
	feature40_dict[f'pseudo_color_{number_to_bind}'] = keys
print(f"Semi-global color features with bind numbers: {numbers_to_bind}")
n_semi_global_color_features = len(keys)
logtxt += f"{n_semi_global_color_features} semi-global color features generated\n"
### Rubin + 7DT
logtxt += "\nRubin Filters + 40 Filters\n"
# 40개 필터의 이름 추출
filter_columns = [f'magobs_{filter}' for filter in MEDIUM_BANDS]

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
rubin_keys = []

color_combinations = list(itertools.combinations(MEDIUM_BANDS, 2))
for f1, f2 in color_combinations:
	color_name = f"{f1}-{f2}"
	# if color_name not in eng_table.keys()
	# 	eng_table[color_name] = photometry_table[f'magobs_{f1}'] - photometry_table[f'magobs_{f2}']
	keys.append(color_name)

for broad_filter in BROAD_BANDS:
    broad_eff = LSST_FILTER_EFF_WAVELENGTHS[broad_filter]
    for med_filter in MEDIUM_BANDS:
        med_eff = float(med_filter[1:])
        
        # 파장 기준으로 정렬
        if broad_eff < med_eff:
            f1, f2 = broad_filter, med_filter
            col1 = photometry_table[f"magobs_{broad_filter}"]
            col2 = photometry_table[f"magobs_{med_filter}"]
        else:
            f1, f2 = med_filter, broad_filter
            col1 = photometry_table[f"magobs_{med_filter}"]
            col2 = photometry_table[f"magobs_{broad_filter}"]
        
        color_name = f"{f1}-{f2}"
        eng_table[color_name] = col1 - col2
        keys.append(color_name)
        rubin_keys.append(color_name)

feature40_dict['rubin_7dt_color'] = keys
feature40_dict['rubin_color'] = rubin_keys
eng_table['uid'] = photometry_table['uid'].data
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature40_dict.keys():
	logtxt += f"\t{key}: {len(feature40_dict[key])}\n"
logtxt += "\n"
### Save
feature40_table_name = os.path.join(FEATURE_NEW_DATA, 'features_40.csv')
print(f"Saving {feature40_table_name}...")
eng_table.write(feature40_table_name, overwrite=True)
print(f"{len(eng_table.keys())-len(basic_keys)} keys generated")
logtxt += f"{len(eng_table.keys())-len(basic_keys)} keys generated\n"
feature40_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_40_color_only.csv')
print(f"Saving {feature40_color_only_table_name}...")
eng_table[basic_keys+feature40_dict['color']].write(feature40_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature40_dict['color'])} keys generated")
logtxt += f"{len(basic_keys+feature40_dict['color'])} color-only keys generated\n"
feature40_rubin_7dt_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_40_rubin_7dt_color_only.csv')
print(f"Saving {feature40_rubin_7dt_color_only_table_name}...")
eng_table[basic_keys+feature40_dict['rubin_7dt_color']].write(feature40_rubin_7dt_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature40_dict['rubin_7dt_color'])} rubin_7dt_color-only keys generated")
logtxt += f"{len(basic_keys+feature40_dict['rubin_7dt_color'])} rubin_7dt_color-only keys generated\n"
feature40_rubin_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_40_rubin_color_only.csv')
print(f"Saving {feature40_rubin_color_only_table_name}...")
eng_table[basic_keys+feature40_dict['rubin_color']].write(feature40_rubin_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature40_dict['rubin_color'])} rubin_color-only keys generated")
logtxt += f"{len(basic_keys+feature40_dict['rubin_color'])} rubin_color-only keys generated\n"
## 20 Features
# %%
# Feature Engineering
logtxt += "\nFeature Engineering\n"
## 20 Features
logtxt += "\n20 Features\n"
feature20_dict = {}
import itertools

# 20개 필터의 이름 추출
filters = MEDIUM_BANDS[::2]
filter_columns = [f'magobs_{filter}' for filter in filters]

# 새로운 피처 테이블 생성
eng20_table = Table()
eng20_table['Sample_ID'] = np.arange(len(photometry_table))
eng20_table['Class'] = photometry_table['type']  # 클래스 열이 있다고 가정
keys = []
color_combinations = list(itertools.combinations(filter_columns, 2))
for f1, f2 in color_combinations:
    color_name = f"{f1.split('_')[1][:4]}-{f2.split('_')[1][:4]}"
    eng20_table[color_name] = photometry_table[f1] - photometry_table[f2]
    keys.append(color_name)
feature20_dict['color'] = keys

n_color_features = len(keys)
logtxt += f"{n_color_features} color features generated\n"
keys = []
for i, filter_col in enumerate(filter_columns):
    if i > 0 and i < len(filter_columns) - 1:
        left = photometry_table[filter_columns[i - 1]]
        right = photometry_table[filter_columns[i + 1]]
        continuum = (left + right) / 2
        residual_name = f"r1_{filter_col.split('_')[1][:4]}"
        eng20_table[residual_name] = photometry_table[filter_col] - continuum
        keys.append(residual_name)
feature20_dict['res1'] = keys

n_res1_features = len(keys)
logtxt += f"{n_res1_features} residual features generated\n"
keys = []
for i, filter_col in enumerate(filter_columns):
    if i > 1 and i < len(filter_columns) - 2:
        left = (photometry_table[filter_columns[i - 2]] + photometry_table[filter_columns[i - 1]]) / 2
        right = (photometry_table[filter_columns[i + 1]] + photometry_table[filter_columns[i + 2]]) / 2
        continuum = (left + right) / 2
        double_residual_name = f"r2_{filter_col.split('_')[1][:4]}"
        eng20_table[double_residual_name] = photometry_table[filter_col] - continuum
        keys.append(double_residual_name)
feature20_dict['res2'] = keys

n_res2_features = len(keys)
logtxt += f"{n_res2_features} double residual features generated\n"

# 모든 필터 묶기와 컬러 생성하기
semi_global_sets = {}
for number_to_bind in numbers_to_bind:
	semi_global_sets[number_to_bind] = add_semi_global_color(filter_columns, number_to_bind, int(number_to_bind / 2))
	#
	keys = []
	for num_filters, columns in semi_global_sets.items():
		color_combinations = itertools.combinations(columns, 2)  # 2개씩 조합
		for (c1_name, c1_value), (c2_name, c2_value) in color_combinations:
			color_name = f"{c1_name}-{c2_name}"
			eng20_table[color_name] = c1_value - c2_value
			keys.append(color_name)
	feature20_dict[f'pseudo_color_{number_to_bind}'] = keys
print(f"Semi-global color features with bind numbers: {numbers_to_bind}")
n_semi_global_color_features = len(keys)
logtxt += f"{n_semi_global_color_features} semi-global color features generated\n"
### Rubin + 7DT
logtxt += "\nRubin Filters + 20 Filters\n"
# 40개 필터의 이름 추출
filter_columns = [f'magobs_{filter}' for filter in filters]

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
rubin_keys = []

color_combinations = list(itertools.combinations(filters, 2))
for f1, f2 in color_combinations:
	color_name = f"{f1}-{f2}"
	# if color_name not in eng_table.keys()
	# 	eng_table[color_name] = photometry_table[f'magobs_{f1}'] - photometry_table[f'magobs_{f2}']
	keys.append(color_name)

for broad_filter in BROAD_BANDS:
    broad_eff = LSST_FILTER_EFF_WAVELENGTHS[broad_filter]
    for med_filter in MEDIUM_BANDS[::2]:
        med_eff = float(med_filter[1:])
        
        # 파장 기준으로 정렬
        if broad_eff < med_eff:
            f1, f2 = broad_filter, med_filter
            col1 = photometry_table[f"magobs_{broad_filter}"]
            col2 = photometry_table[f"magobs_{med_filter}"]
        else:
            f1, f2 = med_filter, broad_filter
            col1 = photometry_table[f"magobs_{med_filter}"]
            col2 = photometry_table[f"magobs_{broad_filter}"]
        
        color_name = f"{f1}-{f2}"
        eng20_table[color_name] = col1 - col2
        keys.append(color_name)
        rubin_keys.append(color_name)


feature20_dict['rubin_7dt_color'] = keys
feature20_dict['rubin_color'] = rubin_keys
eng20_table['uid'] = photometry_table['uid'].data
logtxt += f"{len(eng20_table.keys())-2} keys generated\n"
for key in feature20_dict.keys():
	logtxt += f"\t{key}: {len(feature20_dict[key])}\n"
logtxt += "\n"
eng20_table[:5]
feature20_table_name = os.path.join(FEATURE_NEW_DATA, 'features_20.csv')
print(f"Saving {feature20_table_name}...")
eng20_table.write(feature20_table_name, overwrite=True)
print(f"{len(eng20_table.keys())-2} keys generated")
logtxt += f"{len(eng20_table.keys())-2} keys generated\n"
feature20_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_20_color_only.csv')
print(f"Saving {feature20_color_only_table_name}...")
eng20_table[basic_keys+feature20_dict['color']].write(feature20_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature20_dict['color'])} keys generated")
logtxt += f"{len(basic_keys+feature20_dict['color'])} color-only keys generated\n"
feature20_rubin_7dt_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_20_rubin_7dt_color_only.csv')
print(f"Saving {feature20_rubin_7dt_color_only_table_name}...")
eng20_table[basic_keys+feature20_dict['rubin_7dt_color']].write(feature20_rubin_7dt_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature20_dict['rubin_7dt_color'])} rubin_7dt_color-only keys generated")
logtxt += f"{len(basic_keys+feature20_dict['rubin_7dt_color'])} rubin_7dt_color-only keys generated\n"
feature20_rubin_color_only_table_name = os.path.join(FEATURE_NEW_DATA, 'features_20_rubin_color_only.csv')
print(f"Saving {feature20_rubin_color_only_table_name}...")
eng20_table[basic_keys+feature20_dict['rubin_color']].write(feature20_rubin_color_only_table_name, overwrite=True)
print(f"{len(basic_keys+feature20_dict['rubin_color'])} rubin_color-only keys generated")
logtxt += f"{len(basic_keys+feature20_dict['rubin_color'])} rubin_color-only keys generated\n"
# END
import yaml
with open(os.path.join(CONFIG, 'feature40.yaml'), 'w') as f:
    yaml.dump(feature40_dict, f, default_flow_style=False, allow_unicode=True)

with open(os.path.join(CONFIG, 'feature20.yaml'), 'w') as f:
    yaml.dump(feature20_dict, f, default_flow_style=False, allow_unicode=True)
with open(os.path.join(FEATURE_NEW_DATA, 'log.txt'), 'w') as f:
	f.write(logtxt)
print(logtxt)