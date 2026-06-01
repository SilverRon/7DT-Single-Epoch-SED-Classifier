# 7DT Single-Epoch SED Classifier

A hybrid machine-learning framework for kilonova (KN) anomaly detection using single-epoch medium-band spectral energy distributions (SEDs) from the 7-Dimensional Telescope (7DT).

**Paper:** Gregory S. H. Paek et al. 2026, ApJ, 1001, 198  
**DOI:** [10.3847/1538-4357/ae5229](https://doi.org/10.3847/1538-4357/ae5229)

---

## What this repository provides

| Component | Description |
|-----------|-------------|
| `src/` | Core Python library (model classes, feature utilities, path/variable definitions) |
| `simulator/` | 7DT-Simulator (`sdtpy.py`) — synthetic photometry with realistic noise |
| `script/` | Batch scripts for synphot generation, feature engineering, and model tuning |
| `notebook/` | Full analysis pipeline in Jupyter (data preparation → tuning → evaluation) |
| `config/` | YAML configuration files (feature lists, class labels, model hyperparameters) |
| `model/Tune_XGBoost_{20,40}/` | **Pretrained XGBoost classifiers** (macro-F₁ ≈ 0.80 / 0.84) |
| `model/iForest_{20,40}/` | **Pretrained Isolation Forest anomaly detectors** |
| `data/Feature/Engrave/` | Color-index features of AT 2017gfo (7 VLT/X-shooter epochs) |
| `example_data/` | Small non-KN sample (80 rows, 6 classes) for running the tutorial |
| `tutorial.ipynb` | **Step-by-step tutorial notebook** |

---

## Key results (Paek et al. 2026)

| Metric | 20-filter | 40-filter |
|--------|-----------|-----------|
| Multiclass macro-F₁ | 0.80 | 0.84 |
| KN recall (simulated) | 93% | 93% |
| Non-KN false-positive rate | 9% | 9% |
| AT 2017gfo detection | 6/7 epochs | 6/7 epochs |

The framework classifies sources into 8 non-KN classes (Ia, II, Ibc, SLSN, AGN, TDE, SV, Asteroid) and flags kilonovae as anomalies via a hybrid XGBoost + Isolation Forest decision rule.

---

## Installation

```bash
git clone https://github.com/SilverRon/7DT-Single-Epoch-SED-Classifier.git
cd 7DT-Single-Epoch-SED-Classifier

# Minimum: inference with pretrained models
pip install numpy pandas scikit-learn xgboost joblib pyyaml matplotlib seaborn

# Full: feature engineering, retraining, all notebooks
pip install -r requirements.txt
```

Tested with Python 3.8 (`conda env sedml`). Compatible with Python 3.8–3.12.

---

## Quickstart

```python
import joblib, xgboost as xgb, numpy as np, pandas as pd

# Load pretrained models (20-filter)
xgb_model = joblib.load('model/Tune_XGBoost_20/xgboost_7DT.pkl')
iforest    = joblib.load('model/iForest_20/isolation_forest_base.pkl')

# Load AT 2017gfo features (bundled example data)
df = pd.read_csv('data/Feature/Engrave/features_20_color_only.csv')
feature_cols = xgb_model.feature_names
X = df[feature_cols].fillna(-99)          # NaN = non-detection sentinel

# Step 1: Multiclass classification
proba  = xgb_model.predict(xgb.DMatrix(X))   # shape (n, 8)
p_max  = proba.max(axis=1)

# Step 2: Anomaly score
p_ano  = -iforest.decision_function(X)        # higher = more anomalous

# Step 3: Hybrid KN flag (20-filter thresholds from Youden's J)
kn_flag = ((1 - p_max) > 0.081) & (p_ano > -0.099)
print(kn_flag)
```

---

## Tutorial

Open [`tutorial.ipynb`](tutorial.ipynb) for a step-by-step walkthrough:

1. Load pretrained XGBoost and Isolation Forest models
2. Classify example non-KN transients (6 classes)
3. Score AT 2017gfo (7 epochs of the only confirmed KN)
4. Apply the hybrid AND-gate decision and visualise the 2D decision space
5. Run inference on your own 7DT photometry

---

## Models and data

### Pretrained models (included)

| Path | Description | Size |
|------|-------------|------|
| `model/Tune_XGBoost_20/xgboost_7DT.pkl` | XGBoost, 20 filters (190 features) | 5 MB |
| `model/Tune_XGBoost_40/xgboost_7DT.pkl` | XGBoost, 40 filters (780 features) | 4 MB |
| `model/iForest_20/isolation_forest_base.pkl` | Isolation Forest, 20 filters | 860 KB |
| `model/iForest_40/isolation_forest_base.pkl` | Isolation Forest, 40 filters | 1.4 MB |

### Data not included (large files)

The following were excluded due to size. They are required for retraining and reproducing paper figures.

| Directory | Contents | Size |
|-----------|----------|------|
| `data/raw_data/` | OSC/WISeREP spectra + Wollaeger+21 KN grid | 3.8 GB |
| `data/Spectra/` | Processed spectra and metadata | 3.6 GB |
| `data/Synphot/` | 7DT + Rubin synthetic photometry tables | 7.0 GB |
| `data/Feature/{Original,Augmented,Balanced,New}/` | Full feature tables | ~37 GB |

---

## Repository structure

```
7DT-Single-Epoch-SED-Classifier/
├── config/          YAML configs (features, classes, hyperparameters)
├── data/
│   └── Feature/Engrave/   AT 2017gfo features (bundled)
├── example_data/    Small non-KN sample for tutorial
├── model/
│   ├── Tune_XGBoost_{20,40}/   Final paper XGBoost models
│   ├── iForest_{20,40}/        Final paper iForest models
│   └── Train_Test_Index/       Train/test split indices
├── notebook/        Full Jupyter analysis pipeline
├── references/      Published paper PDF and summary
├── script/          HPC batch scripts
├── simulator/       7DT-Simulator (sdtpy.py)
├── src/             Core library (model.py, helper.py, paths.py, var.py)
├── test/            Diagnostic notebooks
├── tutorial.ipynb   Step-by-step tutorial
└── requirements.txt Python dependencies
```

---

## Citation

If you use this code or the pretrained models, please cite:

```bibtex
@article{Paek2026,
  author  = {Paek, Gregory S. H. and Im, Myungshin and Chang, Seo-Won
             and Choi, Hyeonho and Kim, Ji Hoon},
  title   = {A Hybrid Framework for Kilonova Anomaly Detection Using
              Single-epoch SEDs from the 7-Dimensional Telescope},
  journal = {The Astrophysical Journal},
  year    = {2026},
  volume  = {1001},
  pages   = {198},
  doi     = {10.3847/1538-4357/ae5229}
}
```

---

## Acknowledgments

This research was supported by the National Research Foundation of Korea (NRF) grants funded by the Korean government (MSIT).  
The 7DT data used in this work were acquired at El Sauce Observatory, Chile, as part of the 7-Dimensional Sky Survey (7DS).  
Spectral data were obtained from the Open Supernova Catalog (OSC), WISeREP, and the ENGRAVE Data Release.

---

MIT License — see [LICENSE](LICENSE).
