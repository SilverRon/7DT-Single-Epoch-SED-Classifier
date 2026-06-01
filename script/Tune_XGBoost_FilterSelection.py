#!/usr/bin/env python3
"""
Tune_XGBoost_FilterSelection.py

Train, tune, and evaluate an XGBoost multiclass classifier on 7DT mediumband
SEDs with dynamic filter-set selection.  Configurations are read from a YAML
file; multiple configs are processed sequentially.  For each config, the script:

  1. Derives colour features from the specified filter subset.
  2. Runs Optuna hyperparameter tuning (skipped if best-params file exists).
  3. Trains the final model and saves model, confusion matrix, and report.
  4. Runs 5-fold GroupKFold cross-validation (skipped in --smoke-test mode).

All output is written to <output-dir>/Tune_XGBoost_FilterSel_<config-id>/.

Usage
-----
  # All 26 configurations, CPU
  python Tune_XGBoost_FilterSelection.py

  # Two specific configs on GPU with 8 threads
  python Tune_XGBoost_FilterSelection.py --config-ids no_m500 no_m750_m775 \\
      --device gpu --n-jobs 8

  # Quick sanity check (2 000 rows, 3 Optuna trials, no CV)
  python Tune_XGBoost_FilterSelection.py --smoke-test
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for headless servers
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import joblib
import optuna
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.callback import EarlyStopping

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Matplotlib defaults ───────────────────────────────────────────────────────
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 300
plt.rc("font", family="serif")

# ── Portable path resolution ──────────────────────────────────────────────────
# Works regardless of the directory the script is invoked from.
_SCRIPT_DIR   = Path(__file__).resolve().parent       # …/script/
_PROJECT_ROOT = _SCRIPT_DIR.parent                    # …/SED-Classifier/
_SRC_DIR      = _PROJECT_ROOT / "src"

sys.path.insert(0, str(_SRC_DIR))
from model import plot_confusion_matrix  # seaborn confusion-matrix helper

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG    = _PROJECT_ROOT / "references" / "filter_combinations.yaml"
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "Feature" / "New" / "features_20_color_only.csv"
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "model"

SOURCES_TO_CONSIDER = ["AGN", "Asteroid", "Ia", "II", "Ibc", "SLSN", "SV", "TDE"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_CV_FOLDS   = 5
EARLY_STOP_ROUNDS = 30
SMOKE_TEST_ROWS   = 2_000
SMOKE_TEST_TRIALS = 3


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost prediction helper
# ─────────────────────────────────────────────────────────────────────────────

def _predict_best(model: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    """
    Predict using the early-stopped best iteration when available.

    ``model.best_iteration`` can be 0 (falsy) or even exceed the actual number
    of boosted rounds in some XGBoost versions.  Both cases are handled by
    falling back to plain ``model.predict(dmatrix)``.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        try:
            n_rounds = int(model.num_boosted_rounds())
            if 0 < best_iter <= n_rounds:
                return model.predict(dmatrix, iteration_range=(0, best_iter))
        except Exception:
            pass
    return model.predict(dmatrix)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="PATH",
        help="Filter-combinations YAML  (default: %(default)s)",
    )
    p.add_argument(
        "--config-ids",
        nargs="+",
        metavar="ID",
        default=None,
        help="Run only these config IDs (default: all)",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="XGBoost device  (default: %(default)s)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        metavar="N",
        help="CPU threads for XGBoost; -1 = all cores  (default: %(default)s)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        metavar="N",
        help="Optuna hyperparameter-tuning trials  (default: %(default)s)",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Quick run: {SMOKE_TEST_ROWS} rows, {SMOKE_TEST_TRIALS} Optuna trials, no CV",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=_DEFAULT_DATA_PATH,
        metavar="PATH",
        help="Feature CSV  (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        metavar="PATH",
        help="Base output directory  (default: %(default)s)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────

def load_filter_configs(yaml_path: Path, config_ids: list[str] | None) -> list[dict]:
    """Return filter configurations from YAML, optionally restricted to ``config_ids``."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    all_configs: list[dict] = cfg["configurations"]

    if config_ids is None:
        return all_configs

    id_set = set(config_ids)
    selected = [c for c in all_configs if c["id"] in id_set]
    missing  = id_set - {c["id"] for c in selected}
    if missing:
        raise ValueError(f"Config IDs not found in YAML: {sorted(missing)}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────

def get_color_features(filters: list[str], available_columns: list[str]) -> list[str]:
    """
    Return all pairwise colour feature names (``fA-fB``) derived from ``filters``
    that are present in ``available_columns``.  Order follows the input list.
    """
    available = set(available_columns)
    features = [f"{a}-{b}" for a, b in combinations(filters, 2) if f"{a}-{b}" in available]
    if not features:
        raise ValueError(
            f"No valid colour features found for filters {filters[:3]}… "
            "Check that the filter names match the CSV column names."
        )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: Path, smoke_test: bool) -> pd.DataFrame:
    """
    Read the feature CSV, keep only the target classes, and optionally
    subsample to ``SMOKE_TEST_ROWS`` rows (preserving UID groups).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"[Data] Reading {data_path}")
    data = pd.read_csv(data_path, engine="c")
    data["uid"]   = data["uid"].astype(str)
    data["Class"] = data["Class"].astype(str)

    class_mask = np.zeros(len(data), dtype=bool)
    for cls in SOURCES_TO_CONSIDER:
        class_mask |= (data["Class"] == cls)
    data = data[class_mask].reset_index(drop=True)

    counts = data["Class"].value_counts().to_dict()
    print(f"[Data] {len(data):,} rows  |  classes: { {k: counts.get(k, 0) for k in SOURCES_TO_CONSIDER} }")

    if smoke_test:
        rng  = np.random.RandomState(RANDOM_STATE)
        uids = data["uid"].unique()
        rng.shuffle(uids)
        selected, total = [], 0
        for uid in uids:
            selected.append(uid)
            total += int((data["uid"] == uid).sum())
            if total >= SMOKE_TEST_ROWS:
                break
        data = data[data["uid"].isin(selected)].reset_index(drop=True)
        print(f"[Data] Smoke-test subset: {len(data):,} rows")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning
# ─────────────────────────────────────────────────────────────────────────────

def _build_objective(
    train_dm: xgb.DMatrix,
    test_dm:  xgb.DMatrix,
    n_classes: int,
    y_test_enc: np.ndarray,
    device: str,
    n_jobs: int,
    random_state: int,
):
    """Return a closure that Optuna can call as an objective function."""

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 200, 2000)
        params: dict = {
            "objective":                   "multi:softprob",
            "num_class":                   n_classes,
            "eval_metric":                 "mlogloss",
            "subsample":                   trial.suggest_float("subsample", 0.6, 1.0),
            "min_child_weight":            trial.suggest_int("min_child_weight", 1, 10),
            "max_depth":                   trial.suggest_int("max_depth", 3, 12),
            "learning_rate":               trial.suggest_float("learning_rate", 0.01, 0.3),
            "gamma":                       trial.suggest_float("gamma", 0.0, 0.5),
            "colsample_bytree":            trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":                  trial.suggest_loguniform("reg_lambda", 1e-3, 100),
            "random_state":                random_state,
            "tree_method":                 "hist",
            "device":                      device,
            "nthread":                     n_jobs,
            "disable_default_eval_metric": 1,
        }
        es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name="mlogloss", data_name="validation")
        model = xgb.train(
            params, train_dm,
            num_boost_round=n_estimators,
            evals=[(test_dm, "validation")],
            callbacks=[es],
            verbose_eval=False,
        )
        y_pred = np.argmax(_predict_best(model, test_dm), axis=1)
        score  = f1_score(y_test_enc, y_pred, average="macro")
        del model
        gc.collect()
        return score

    return objective


def run_optuna_tuning(
    train_dm:         xgb.DMatrix,
    test_dm:          xgb.DMatrix,
    n_classes:        int,
    y_test_enc:       np.ndarray,
    device:           str,
    n_jobs:           int,
    n_trials:         int,
    random_state:     int,
    best_params_path: Path,
) -> dict:
    """
    Run (or reload) an Optuna study and return the best hyperparameters.
    If ``best_params_path`` already exists the study is skipped and the saved
    params are loaded instead.
    """
    if best_params_path.exists():
        print(f"[Tune] Loading existing params: {best_params_path.name}")
        with open(best_params_path) as f:
            return yaml.safe_load(f)

    print(f"[Tune] Starting Optuna ({n_trials} trials, device={device})")
    t0    = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _build_objective(train_dm, test_dm, n_classes, y_test_enc, device, n_jobs, random_state),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    best    = study.best_trial
    print(f"[Tune] Finished in {elapsed/60:.1f} min  |  best macro-F1 = {best.value:.4f}")

    best_params: dict = dict(best.params)
    best_params["random_state"] = random_state
    best_params["nthread"]      = n_jobs
    if device == "gpu":
        best_params["device"] = device

    with open(best_params_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"[Tune] Best params → {best_params_path}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build fixed XGBoost params from Optuna output
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_xgb_params(raw_params: dict, n_classes: int, device: str, n_jobs: int) -> tuple[dict, int]:
    """
    Return ``(params_dict, num_boost_round)`` suitable for ``xgb.train``.

    Pops ``n_estimators`` (Optuna key) and injects the fixed XGBoost keys
    that must not be changed between tuning and final training.  The input
    dict is not mutated.
    """
    params = raw_params.copy()
    num_boost_round: int = params.pop("n_estimators", 2000)
    params.update(
        {
            "objective":                   "multi:softprob",
            "num_class":                   n_classes,
            "eval_metric":                 "mlogloss",
            "disable_default_eval_metric": 1,
            "tree_method":                 "hist",
            "device":                      device,
            "nthread":                     n_jobs,
        }
    )
    return params, num_boost_round


# ─────────────────────────────────────────────────────────────────────────────
# Final training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    best_params:  dict,
    train_dm:     xgb.DMatrix,
    test_dm:      xgb.DMatrix,
    y_test_enc:   np.ndarray,
    class_names:  np.ndarray,
    n_trials:     int,
    device:       str,
    n_jobs:       int,
    path_save:    Path,
) -> bool:
    """
    Train the final model with ``best_params``; persist model, classification
    report, and confusion matrix.  Returns ``True`` if training was performed,
    ``False`` if outputs already existed and were skipped.
    """
    cm_path      = path_save / f"confusion_matrix_n{n_trials}.csv"
    report_path  = path_save / f"classification_report_n{n_trials}.csv"
    model_path   = path_save / f"xgboost_7DT_n{n_trials}.pkl"
    cm_png_path  = path_save / f"confusion_matrix_n{n_trials}.png"

    if cm_path.exists():
        print(f"[Train] Outputs exist, skipping training: {cm_path.name}")
        return False

    print("[Train] Training final model")
    params, num_boost_round = _prepare_xgb_params(best_params, len(class_names), device, n_jobs)

    es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name="mlogloss", data_name="validation")
    model = xgb.train(
        params, train_dm,
        num_boost_round=num_boost_round,
        evals=[(test_dm, "validation")],
        callbacks=[es],
        verbose_eval=False,
    )

    y_pred_proba = _predict_best(model, test_dm)
    y_pred_enc   = np.argmax(y_pred_proba, axis=1)

    joblib.dump(model, model_path)
    print(f"[Train] Model saved → {model_path.name}")

    # ── Classification report ─────────────────────────────────────────────────
    report_dict = classification_report(
        y_test_enc, y_pred_enc,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    acc = report_dict.pop("accuracy")
    report_df = pd.DataFrame(report_dict).T
    report_df.loc["accuracy"] = [None, None, acc, report_df["support"].sum()]
    report_df.to_csv(report_path)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_test_enc, y_pred_enc, class_names)
    plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    cm = confusion_matrix(y_test_enc, y_pred_enc)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)

    macro_f1 = f1_score(y_test_enc, y_pred_enc, average="macro")
    print(f"[Train] Test macro-F1 = {macro_f1:.4f}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    all_dm:      xgb.DMatrix,
    y_enc:       np.ndarray,
    full_X:      pd.DataFrame,
    uids:        np.ndarray,
    best_params: dict,
    n_classes:   int,
    class_names: np.ndarray,
    n_trials:    int,
    device:      str,
    n_jobs:      int,
    path_save:   Path,
) -> None:
    """Run 5-fold GroupKFold CV; write per-fold macro-F1 and per-class F1 CSVs."""
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    print(f"[CV] Running {N_CV_FOLDS}-fold GroupKFold cross-validation")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(full_X, y_enc, groups=uids)):
        macro_path     = path_save / f"cv_macro_f1_scores_n{n_trials}_{fold}.csv"
        per_class_path = path_save / f"cv_per_class_f1_scores_n{n_trials}_{fold}.csv"

        if macro_path.exists() and per_class_path.exists():
            print(f"[CV] Fold {fold} exists, skipping")
            continue

        params, num_boost_round = _prepare_xgb_params(best_params, n_classes, device, n_jobs)
        train_dm = all_dm.slice(train_idx)
        val_dm   = all_dm.slice(val_idx)

        es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name="mlogloss", data_name="validation")
        model_cv = xgb.train(
            params, train_dm,
            num_boost_round=num_boost_round,
            evals=[(val_dm, "validation")],
            callbacks=[es],
            verbose_eval=False,
        )

        y_pred_proba = _predict_best(model_cv, val_dm)
        y_pred_enc   = np.argmax(y_pred_proba, axis=1)
        y_val_enc    = val_dm.get_label().astype(int)

        macro_f1 = f1_score(y_val_enc, y_pred_enc, average="macro")
        print(f"[CV] Fold {fold}  macro-F1 = {macro_f1:.4f}")

        pd.DataFrame([{"fold": fold, "f1_macro": macro_f1}]).to_csv(macro_path, index=False)

        report = classification_report(
            y_val_enc, y_pred_enc,
            labels=np.arange(n_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        per_class_f1 = {name: report[name]["f1-score"] for name in class_names}
        per_class_f1["fold"] = fold
        pd.DataFrame([per_class_f1]).to_csv(per_class_path, index=False)

        del model_cv
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Single-config orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_config(cfg: dict, data: pd.DataFrame, args: argparse.Namespace) -> None:
    """Execute the full pipeline for one filter configuration."""
    config_id = cfg["id"]
    filters   = cfg["filters"]
    desc      = cfg.get("description", "")
    print(f"\n{'='*64}")
    print(f"  Config : {config_id}")
    print(f"  Desc   : {desc}")
    print(f"  Filters: {len(filters)}  →  {filters}")
    print(f"{'='*64}")

    # ── Output directory ──────────────────────────────────────────────────────
    path_save = args.output_dir / f"Tune_XGBoost_FilterSel_{config_id}"
    path_save.mkdir(parents=True, exist_ok=True)

    # ── Derive colour features ────────────────────────────────────────────────
    avail_cols      = data.columns.tolist()
    features_to_use = get_color_features(filters, avail_cols)
    print(f"[Config] {len(features_to_use)} colour features")

    # ── Prepare arrays ────────────────────────────────────────────────────────
    # Drop metadata columns; fill nondetections (NaN) with sentinel value -99
    meta_cols = [c for c in ["Sample_ID", "Class", "uid"] if c in data.columns]
    X    = data.drop(columns=meta_cols).fillna(-99)
    y    = data["Class"]
    uids = data["uid"].values

    le         = LabelEncoder()
    y_enc      = le.fit_transform(y)
    class_names = np.array([str(c) for c in le.classes_])
    n_classes   = len(class_names)
    print(f"[Config] Classes ({n_classes}): {class_names}")

    # ── Train / test split (group-aware, so entire UIDs stay in one split) ────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=uids))

    X_train = X.iloc[train_idx][features_to_use]
    X_test  = X.iloc[test_idx][features_to_use]
    y_train_enc = le.transform(y.iloc[train_idx])
    y_test_enc  = le.transform(y.iloc[test_idx])

    train_dm = xgb.DMatrix(X_train, label=y_train_enc)
    test_dm  = xgb.DMatrix(X_test,  label=y_test_enc)

    n_trials = SMOKE_TEST_TRIALS if args.smoke_test else args.n_trials

    # ── 1. Hyperparameter tuning ──────────────────────────────────────────────
    best_params_path = path_save / f"best_params_n{n_trials}.yaml"
    best_params = run_optuna_tuning(
        train_dm, test_dm,
        n_classes=n_classes,
        y_test_enc=y_test_enc,
        device=args.device,
        n_jobs=args.n_jobs,
        n_trials=n_trials,
        random_state=RANDOM_STATE,
        best_params_path=best_params_path,
    )

    # Overwrite device/thread from CLI so a model tuned on GPU can be evaluated
    # on CPU (or vice-versa) without re-tuning.
    best_params["nthread"] = args.n_jobs
    best_params["device"]  = args.device

    # ── 2. Final training & evaluation ────────────────────────────────────────
    train_and_evaluate(
        best_params, train_dm, test_dm,
        y_test_enc, class_names,
        n_trials, args.device, args.n_jobs, path_save,
    )

    # ── 3. Cross-validation (skipped in smoke-test mode) ─────────────────────
    if not args.smoke_test:
        sub_X  = X[features_to_use]
        all_dm = xgb.DMatrix(sub_X, label=y_enc)
        run_cross_validation(
            all_dm, y_enc, X, uids,
            best_params,
            n_classes=n_classes,
            class_names=class_names,
            n_trials=n_trials,
            device=args.device,
            n_jobs=args.n_jobs,
            path_save=path_save,
        )

    print(f"[Done] {config_id} → {path_save}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"[Main] Project root : {_PROJECT_ROOT}")
    print(f"[Main] Config YAML  : {args.config}")
    print(f"[Main] Data         : {args.data_path}")
    print(f"[Main] Output dir   : {args.output_dir}")
    print(f"[Main] Device       : {args.device}  |  n_jobs={args.n_jobs}  |  n_trials={args.n_trials}")
    if args.smoke_test:
        print(f"[Main] *** SMOKE-TEST MODE: {SMOKE_TEST_ROWS} rows, {SMOKE_TEST_TRIALS} trials, no CV ***")

    # Load all (or selected) filter configurations
    configs = load_filter_configs(args.config, args.config_ids)
    print(f"[Main] {len(configs)} configuration(s) to process\n")

    # Load feature data once — shared across all configs
    data = load_data(args.data_path, args.smoke_test)

    # Process each configuration sequentially
    n_ok, n_err = 0, 0
    for cfg in configs:
        try:
            run_config(cfg, data, args)
            n_ok += 1
        except Exception:
            print(f"\n[ERROR] Config '{cfg['id']}' failed:")
            traceback.print_exc()
            n_err += 1

    print(f"\n[Main] Complete — {n_ok} succeeded, {n_err} failed.")


if __name__ == "__main__":
    main()
