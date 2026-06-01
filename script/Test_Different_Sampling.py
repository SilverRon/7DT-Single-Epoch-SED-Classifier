# %%
# 1. Common Data Preprocessing and Splitting
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

# Matplotlib global settings
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# ML libraries
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Helper functions & model import
sys.path.append(os.path.join('..', 'src'))
from helper import makeSpecColors
from paths import *
from var import *
from sdtpy import *
from model import (
	#	Sampling
	sample_by_uid_group,
	#	Model Experiments
    experiment_lightgbm,
    # experiment_rf,
    # experiment_xgb,
    # experiment_catboost,
    # experiment_tabnet,
    # experiment_mlp,
)

# %%
# Set experiment configs
test_name = "different_sampling"
random_state = 42
test_size = 0.2
device_type = "cpu" # or gpu
n_jobs = 10
path_save = os.path.join(MODEL, test_name)
os.makedirs(path_save, exist_ok=True)
# %%

sources_to_consider = [
	"AGN", 
	"Ia", 
	"II", 
	"Ibc", 
	"LBV", 
	"TDE", 
	"Nova", 
	"M dwarf", 
	"CV"
]
logtxt += f"\nSources to consider: {sources_to_consider}\n"


metrics_list = []

# %%
# Original Data Set (no augmentation)
path_data = os.path.join(FEATURE_ORIGINAL_DATA, 'features_40.csv')
path_data_augmented = os.path.join(FEATURE_AUGMENTED_DATA, 'features_40.csv')
logtxt += f"\nOriginal Data Set (no augmentation)\n"
# %%
# 1. Load and Concatenate Data
data = pd.read_csv(path_data)
data_augmented = pd.read_csv(path_data_augmented)
# data = pd.concat([data, data_augmented], ignore_index=True)

# Stratified sampling by class, based on uid
# n = 1000  # 원하는 샘플 수 (필요시 수정)
# data = data.groupby('Class', group_keys=False)\
#            .apply(lambda x: sample_by_uid_group(x, n=n, uid_col='uid', random_state=42))\
#            .reset_index(drop=True)
# data = data.dropna(subset=['Class'])

indx_type_to_consider = np.where(
	np.array([(data['Class'] == source) for source in sources_to_consider]).any(axis=0)
)
print(f"{len(sources_to_consider)} sources to consider: {len(indx_type_to_consider[0])}")
data = data.iloc[indx_type_to_consider[0]]
data_augmented = data_augmented.iloc[indx_type_to_consider[0]]

# %%
# Split features/target, handle missing values
X = data.drop(columns=['Sample_ID', 'Class', 'uid'])
y = data['Class']
X.fillna(-99, inplace=True)

X_augmented = data_augmented.drop(columns=['Sample_ID', 'Class', 'uid'])
y_augmented = data_augmented['Class']
X_augmented.fillna(-99, inplace=True)

# %%
# Split into train/test using GroupShuffleSplit by uid
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
train_idx, test_idx = next(gss.split(X, y, groups=data['uid']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_idx_augmented, test_idx_augmented = next(gss.split(X_augmented, y_augmented, groups=data_augmented['uid']))
X_train_augmented, X_test_augmented = X_augmented.iloc[train_idx_augmented], X_augmented.iloc[test_idx_augmented]
y_train_augmented, y_test_augmented = y_augmented.iloc[train_idx_augmented], y_augmented.iloc[test_idx_augmented]

# Label encode class for ML
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
print("Class mapping:", label_encoder.inverse_transform(np.arange(len(label_encoder.classes_))))

y_train_augmented = label_encoder.transform(y_train_augmented)
y_test_augmented = label_encoder.transform(y_test_augmented)
print("Class mapping:", label_encoder.inverse_transform(np.arange(len(label_encoder.classes_))))

# %%
# ----------------------------------------------------
# Model Parameters
# ----------------------------------------------------
classifier_type = 'normal_class_classifier'
model_param_config = model_config[classifier_type][device_type]

# 예시: 각 모델 파라미터 딕셔너리로 할당
params_lightgbm = model_param_config['params_lightgbm']
# %%
# ----------------------------------------------------
# 실험 실행부: 각 모델별 함수 개별 호출 및 결과 저장
# ----------------------------------------------------
eval_metrics_list = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro", "accuracy"]

# %%
# LightGBM
_, lgbm_metrics = experiment_lightgbm(X_train, X_test, y_train, y_test, data, label_encoder, params_lightgbm, eval_metrics_list, path_save, do_cv=False)
metrics_list.append(lgbm_metrics)
# %%

_, lgbm_metrics_augmented = experiment_lightgbm(X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented, data_augmented, label_encoder, params_lightgbm, eval_metrics_list, path_save, do_cv=False)
metrics_list.append(lgbm_metrics_augmented)
# %%


# %%
# ----------------------------------------------------
# 모든 결과를 하나의 csv로 저장
metrics_all = pd.concat(metrics_list, ignore_index=True)
# metrics_all.to_csv(os.path.join(path_save, "metrics_summary.csv"), index=False)
# print("Experiment complete. Results saved in:", path_save)