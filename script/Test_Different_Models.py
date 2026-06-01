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
    experiment_rf,
    experiment_xgb,
    experiment_catboost,
    experiment_tabnet,
    experiment_mlp,
)

# --- Helper: Sample by uid group to ensure group-level stratification ---


# %%
# Set experiment configs
test_name = "different_models"
random_state = 42
test_size = 0.2
device_type = "cpu" # or gpu
n_jobs = 10
path_data = os.path.join(FEATURE_DATA, 'features_40.csv')
path_save = os.path.join(MODEL, test_name)
os.makedirs(path_save, exist_ok=True)

# 1. Load and Concatenate Data
data = pd.read_csv(path_data)

# Stratified sampling by class, based on uid
# n = 1000  # 원하는 샘플 수 (필요시 수정)
# data = data.groupby('Class', group_keys=False)\
#            .apply(lambda x: sample_by_uid_group(x, n=n, uid_col='uid', random_state=42))\
#            .reset_index(drop=True)
# data = data.dropna(subset=['Class'])

# Split features/target, handle missing values
X = data.drop(columns=['Sample_ID', 'Class', 'uid'])
y = data['Class']
X.fillna(-99, inplace=True)

# Split into train/test using GroupShuffleSplit by uid
gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
train_idx, test_idx = next(gss.split(X, y, groups=data['uid']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Label encode class for ML
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
print("Class mapping:", label_encoder.inverse_transform(np.arange(len(label_encoder.classes_))))
# %%
# ----------------------------------------------------
# Model Parameters
# ----------------------------------------------------
classifier_type = 'normal_class_classifier'
model_param_config = model_config[classifier_type][device_type]

# 예시: 각 모델 파라미터 딕셔너리로 할당
params_lightgbm = model_param_config['params_lightgbm']
params_rf = model_param_config['params_rf']
params_xgb = model_param_config['params_xgb']
params_catboost = model_param_config['params_catboost']
params_tabnet = model_param_config['params_tabnet']
params_mlp = model_param_config['params_mlp']
# %%
# ----------------------------------------------------
# 실험 실행부: 각 모델별 함수 개별 호출 및 결과 저장
# ----------------------------------------------------
eval_metrics_list = ["f1_macro", "f1_weighted", "precision_macro", "recall_macro", "accuracy"]

metrics_list = []
# %%
# LightGBM
_, lgbm_metrics = experiment_lightgbm(X_train, X_test, y_train, y_test, data, label_encoder, params_lightgbm, eval_metrics_list, path_save)
metrics_list.append(lgbm_metrics)
# %%
# Random Forest
_, rf_metrics = experiment_rf(X_train, X_test, y_train, y_test, data, label_encoder, params_rf, eval_metrics_list, path_save)
metrics_list.append(rf_metrics)
# %%
# XGBoost
_, xgb_metrics = experiment_xgb(X_train, X_test, y_train, y_test, data, label_encoder, params_xgb, eval_metrics_list, path_save)
metrics_list.append(xgb_metrics)
# %%
# CatBoost
_, cat_metrics = experiment_catboost(X_train, X_test, y_train, y_test, data, label_encoder, params_catboost, eval_metrics_list, path_save)
metrics_list.append(cat_metrics)
# %%
# TabNet
_, tabnet_metrics = experiment_tabnet(X_train, X_test, y_train, y_test, data, label_encoder, params_tabnet, eval_metrics_list, path_save)
metrics_list.append(tabnet_metrics)
# %%
# MLP
_, mlp_metrics = experiment_mlp(X_train, X_test, y_train, y_test, data, label_encoder, params_mlp, eval_metrics_list, path_save)
metrics_list.append(mlp_metrics)

# ----------------------------------------------------
# 모든 결과를 하나의 csv로 저장
metrics_all = pd.concat(metrics_list, ignore_index=True)
metrics_all.to_csv(os.path.join(path_save, "metrics_summary.csv"), index=False)
print("Experiment complete. Results saved in:", path_save)