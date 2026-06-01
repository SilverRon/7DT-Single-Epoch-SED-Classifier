# Library
# %%
# Python Library
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
# %%
logtxt = ""
# %%
# Data
photometry_table_file = os.path.join(AUGMNETED_PHOT_DATA, "merged_synphot_normal_class.csv")
photometry_table = Table.read(photometry_table_file)
print(f"{len(photometry_table):,} photometry table read")
logtxt += f"{len(photometry_table):,} photometry table read\n"
# %%
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

# 1. Feature 1: 40개 필터의 측광값 추가 --> Skip
# keys = []
# for filter_col in filter_columns:
#     newkey = filter_col.split('_')[1][:4]
#     eng_table[newkey] = photometry_table[filter_col]
#     keys.append(newkey)
# feature40_dict['phot'] = keys

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
color_combinations = list(itertools.combinations(filter_columns, 2))
for f1, f2 in color_combinations:
    color_name = f"{f1.split('_')[1][:4]}-{f2.split('_')[1][:4]}"
    eng_table[color_name] = photometry_table[f1] - photometry_table[f2]
    keys.append(color_name)
feature40_dict['color'] = keys

n_color_features = len(keys)
logtxt += f"{n_color_features} color features generated\n"

# 3. Feature 3: 양 옆의 필터로 생성된 residual 값
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

# 4. Feature 4: 양 옆 두 개의 필터로 생성된 residual 값
# keys = []
# for i, filter_col in enumerate(filter_columns):
#     if i > 1 and i < len(filter_columns) - 2:
#         left = (photometry_table[filter_columns[i - 2]] + photometry_table[filter_columns[i - 1]]) / 2
#         right = (photometry_table[filter_columns[i + 1]] + photometry_table[filter_columns[i + 2]]) / 2
#         continuum = (left + right) / 2
#         double_residual_name = f"r2_{filter_col.split('_')[1][:4]}"
#         eng_table[double_residual_name] = photometry_table[filter_col] - continuum
#         keys.append(double_residual_name)
# feature40_dict['res2'] = keys

# 5. Semi-global color feature 추가: 다양한 간격으로 연속적인 필터 묶기
# 테이블에 추가하지 않고 리스트에만 저장
def add_semi_global_color(filter_columns, num_filters, interval):
    filters = []
    for i in range(0, len(filter_columns) - num_filters + 1, interval):
        subset = filter_columns[i:i + num_filters]
        semi_global_name = f"c{num_filters:g}_{subset[0].split('_')[1][0:4]}_{subset[-1].split('_')[1][1:4]}"
        semi_global_value = sum(photometry_table[filt] for filt in subset) / num_filters
        filters.append((semi_global_name, semi_global_value))
    return filters

# 모든 필터 묶기와 컬러 생성하기
semi_global_sets = {}
numbers_to_bind = [2, 4, 8, 10, 20]
for number_to_bind in numbers_to_bind:
    semi_global_sets[number_to_bind] = add_semi_global_color(filter_columns, number_to_bind, int(number_to_bind / 2))

# Semi-global 컬러 조합 생성 - 컬러 조합만 테이블에 추가
keys = []
for num_filters, columns in semi_global_sets.items():
    color_combinations = itertools.combinations(columns, 2)  # 2개씩 조합
    for (c1_name, c1_value), (c2_name, c2_value) in color_combinations:
        color_name = f"{c1_name}-{c2_name}"
        eng_table[color_name] = c1_value - c2_value
        keys.append(color_name)
feature40_dict['semi_global_color'] = keys

print(f"Semi-global color features with bind numbers: {numbers_to_bind}")
n_semi_global_color_features = len(keys)
logtxt += f"{n_semi_global_color_features} semi-global color features generated\n"

# 최종 테이블 저장
eng_table['uid'] = photometry_table['uid'].data
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature40_dict.keys():
	logtxt += f"\t{key}: {len(feature40_dict[key])}\n"
logtxt += "\n"

feature40_table_name = os.path.join(FEATURE_AUGMENTED_DATA, 'features_40.csv')
eng_table.write(feature40_table_name, overwrite=True)
print(f"{len(eng_table.keys())-2} keys generated")
logtxt += f"{len(eng_table.keys())-2} keys generated\n"

# %%
# 20 Filters = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# 1. dictionary를 JSON 파일로 저장하기
# with open(f'{path_save}/feature_group_40.json', 'w') as json_file:
#     json.dump(feature40_dict, json_file)

# 2. JSON 파일을 열어서 dictionary로 다시 로드하기
# with open('my_data.json', 'r') as json_file:
#     loaded_dict = json.load(json_file)
## 20 Features
filters = MEDIUM_BANDS[::2]
not_20_filters = [filte for filte in MEDIUM_BANDS if filte not in filters]

columns_for_20filters = ['type', 'spec', 'flagtype']
for filte in filters:
	columns_for_20filters.append(f'magobs_{filte}')

columns_for_20filters

# %%
logtxt += f"\n\n20 Filters\n"
feature20_table = photometry_table[columns_for_20filters]
feature20_table[:3]
feature20_dict = {}

# 새로운 피처 테이블 생성
eng_table = Table()
eng_table['Sample_ID'] = np.arange(len(feature20_table))
eng_table['Class'] = feature20_table['type']  # 클래스 열이 있다고 가정

filter_columns = columns_for_20filters.copy()
filter_columns.remove('type')
filter_columns.remove('spec')
filter_columns.remove('flagtype')

logtxt += f"{len(filter_columns)} filters selected\n"

# 1. Feature 1: 20개 필터의 측광값 추가 --> Skip
# keys = []
# for filter_col in columns_for_20filters:
#     newkey = filter_col.split('_')[1][:4]
#     eng_table[newkey] = feature20_table[filter_col]
#     keys.append(newkey)
# feature20_dict['phot'] = keys

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
color_combinations = list(itertools.combinations(filter_columns, 2))
for f1, f2 in color_combinations:
    color_name = f"{f1.split('_')[1][:4]}-{f2.split('_')[1][:4]}"
    eng_table[color_name] = feature20_table[f1] - feature20_table[f2]
    keys.append(color_name)
feature20_dict['color'] = keys
logtxt += f"{len(keys)} color features generated\n"

# 3. Feature 3: 양 옆의 필터로 생성된 residual 값
keys = []
for i, filter_col in enumerate(filter_columns):
    if i > 0 and i < len(filter_columns) - 1:
        left = feature20_table[filter_columns[i - 1]]
        right = feature20_table[filter_columns[i + 1]]
        continuum = (left + right) / 2
        residual_name = f"r1_{filter_col.split('_')[1][:4]}"
        eng_table[residual_name] = feature20_table[filter_col] - continuum
        keys.append(residual_name)
feature20_dict['res1'] = keys
logtxt += f"{len(keys)} residual features generated\n"

# 4. Feature 4: 양 옆 두 개의 필터로 생성된 residual 값
# keys = []
# for i, filter_col in enumerate(filter_columns):
#     if i > 1 and i < len(filter_columns) - 2:
#         left = (feature20_table[filter_columns[i - 2]] + feature20_table[filter_columns[i - 1]]) / 2
#         right = (feature20_table[filter_columns[i + 1]] + feature20_table[filter_columns[i + 2]]) / 2
#         continuum = (left + right) / 2
#         double_residual_name = f"r2_{filter_col.split('_')[1][:4]}"
#         eng_table[double_residual_name] = feature20_table[filter_col] - continuum
#         keys.append(double_residual_name)
# feature20_dict['res2'] = keys

# 5. Semi-global color feature 추가: 다양한 간격으로 연속적인 필터 묶기
# 테이블에 추가하지 않고 리스트에만 저장
def add_semi_global_color(filter_columns, num_filters, interval):
    filters = []
    for i in range(0, len(filter_columns) - num_filters + 1, interval):
        subset = filter_columns[i:i + num_filters]
        semi_global_name = f"c{num_filters:g}_{subset[0].split('_')[1][0:4]}_{subset[-1].split('_')[1][1:4]}"
        semi_global_value = sum(feature20_table[filt] for filt in subset) / num_filters
        filters.append((semi_global_name, semi_global_value))
    return filters

# 모든 필터 묶기와 컬러 생성하기
semi_global_sets = {}
for number_to_bind in [2, 4, 10]:
    semi_global_sets[number_to_bind] = add_semi_global_color(filter_columns, number_to_bind, int(number_to_bind / 2))

# Semi-global 컬러 조합 생성 - 컬러 조합만 테이블에 추가
keys = []
for num_filters, columns in semi_global_sets.items():
    color_combinations = itertools.combinations(columns, 2)  # 2개씩 조합
    for (c1_name, c1_value), (c2_name, c2_value) in color_combinations:
        color_name = f"{c1_name}-{c2_name}"
        eng_table[color_name] = c1_value - c2_value
        keys.append(color_name)
feature20_dict['semi_global_color'] = keys
logtxt += f"{len(keys)} semi-global color features generated\n"

# 최종 테이블 저장
# eng_table.write(f"{path_save}/feature_eng.csv", overwrite=True)
eng_table['uid'] = photometry_table['uid'].data
feature20_table_name = os.path.join(FEATURE_AUGMENTED_DATA, 'features_20.csv')
eng_table.write(feature20_table_name, overwrite=True)
print(f"{len(eng_table.keys())-2} keys generated")
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature20_dict.keys():
	logtxt += f"\t{key}: {len(feature20_dict[key])}\n"
logtxt += "\n"
# 1. dictionary를 JSON 파일로 저장하기
# with open(f'{path_save}/feature_group_20.json', 'w') as json_file:
#     json.dump(feature20_dict, json_file)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Rubin Filters
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# %%
logtxt += "\nRubin Filters + 40 Filters\n"
feature_rubin_7dt40_dict = {}
# 40개 필터의 이름 추출
filter_columns = [f'magobs_{filter}' for filter in MEDIUM_BANDS]

# 새로운 피처 테이블 생성
eng_table = Table()
eng_table['Sample_ID'] = np.arange(len(photometry_table))
eng_table['Class'] = photometry_table['type']  # 클래스 열이 있다고 가정

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
color_combinations = list(itertools.combinations(MEDIUM_BANDS, 2))
for f1, f2 in color_combinations:
    color_name = f"{f1}-{f2}"
    eng_table[color_name] = photometry_table[f'magobs_{f1}'] - photometry_table[f'magobs_{f2}']
    keys.append(color_name)

for broad_filter in BROAD_BANDS:
	eff_wavelength = LSST_FILTER_EFF_WAVELENGTHS[broad_filter]
	for med_filter in MEDIUM_BANDS:
		med_eff_wavelength = float(med_filter[1:])
		if med_eff_wavelength > eff_wavelength:
			color_name = f"{broad_filter}-{med_filter}"
			eng_table[color_name] = photometry_table[f'magobs_{broad_filter}'] - photometry_table[f'magobs_{med_filter}']
		else:
			color_name = f"{med_filter}-{broad_filter}"
			eng_table[color_name] = photometry_table[f'magobs_{med_filter}'] - photometry_table[f'magobs_{broad_filter}']
		keys.append(color_name)

feature_rubin_7dt40_dict['color'] = keys

eng_table['uid'] = photometry_table['uid'].data
feature_rubin_7dt40_table_name = os.path.join(FEATURE_AUGMENTED_DATA, 'features_rubin_7dt40.csv')
eng_table.write(feature_rubin_7dt40_table_name, overwrite=True)
print(f"{len(eng_table.keys())-2} keys generated")
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature_rubin_7dt40_dict.keys():
	logtxt += f"\t{key}: {len(feature_rubin_7dt40_dict[key])}\n"
logtxt += "\n"

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Rubin + 20 filters
logtxt += "\nRubin Filters + 20 Filters\n"
feature_rubin_7dt20_dict = {}

eng_table['uid'] = photometry_table['uid'].data
feature_rubin_7dt20_table_name = os.path.join(FEATURE_AUGMENTED_DATA, 'features_rubin_7dt20.csv')
eng_table.write(feature_rubin_7dt20_table_name, overwrite=True)
print(f"{len(eng_table.keys())-2} keys generated")
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature_rubin_7dt20_dict.keys():
	logtxt += f"\t{key}: {len(feature_rubin_7dt20_dict[key])}\n"
logtxt += "\n"

logtxt += "\nRubin Filters + 20 Filters\n"
feature_rubin_7dt20_dict = {}
# 20개 필터의 이름 추출
filter_columns = [f'magobs_{filter}' for filter in MEDIUM_BANDS[::2]]

# 새로운 피처 테이블 생성
eng_table = Table()
eng_table['Sample_ID'] = np.arange(len(photometry_table))
eng_table['Class'] = photometry_table['type']  # 클래스 열이 있다고 가정

# 2. Feature 2: 모든 필터 조합의 색상 (e.g., m400 - m650)
keys = []
color_combinations = list(itertools.combinations(MEDIUM_BANDS[::2], 2))
for f1, f2 in color_combinations:
    color_name = f"{f1}-{f2}"
    eng_table[color_name] = photometry_table[f'magobs_{f1}'] - photometry_table[f'magobs_{f2}']
    keys.append(color_name)

for broad_filter in BROAD_BANDS:
	eff_wavelength = LSST_FILTER_EFF_WAVELENGTHS[broad_filter]
	for med_filter in MEDIUM_BANDS[::2]:
		med_eff_wavelength = float(med_filter[1:])
		if med_eff_wavelength > eff_wavelength:
			color_name = f"{broad_filter}-{med_filter}"
			eng_table[color_name] = photometry_table[f'magobs_{broad_filter}'] - photometry_table[f'magobs_{med_filter}']
		else:
			color_name = f"{med_filter}-{broad_filter}"
			eng_table[color_name] = photometry_table[f'magobs_{med_filter}'] - photometry_table[f'magobs_{broad_filter}']
		keys.append(color_name)

feature_rubin_7dt20_dict['color'] = keys

eng_table['uid'] = photometry_table['uid'].data
feature_rubin_7dt20_table_name = os.path.join(FEATURE_AUGMENTED_DATA, 'features_rubin_7dt20.csv')
eng_table.write(feature_rubin_7dt20_table_name, overwrite=True)
print(f"{len(eng_table.keys())-2} keys generated")
logtxt += f"{len(eng_table.keys())-2} keys generated\n"
for key in feature_rubin_7dt20_dict.keys():
	logtxt += f"\t{key}: {len(feature_rubin_7dt20_dict[key])}\n"
logtxt += "\n"


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# %%
feature_information = {
	'7DT_40': feature40_dict,
	'7DT_20': feature20_dict,
    'Rubin+7DT_40': feature_rubin_7dt40_dict,
    'Rubin+7DT_20': feature_rubin_7dt20_dict,
}

# %%
import yaml
# 저장
feature_config_name = os.path.join(CONFIG, 'feature.yaml')
with open(feature_config_name, 'w') as f:
    yaml.dump(feature_information, f, default_flow_style=False, allow_unicode=True)
total_number = 0

for key, val in feature_information.items():
	print(f"{key}: {len(val)}")
	total_number += len(val)
print(total_number)
classes = np.unique(eng_table['Class'])
logtxt += f"Classes: {classes}\n"

logtxt += "END\n"
logtxt_path = os.path.join(FEATURE_AUGMENTED_DATA, "log.txt")
with open(logtxt_path, 'w') as f:
    f.write(logtxt)

