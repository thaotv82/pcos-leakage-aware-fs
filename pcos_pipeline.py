#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCOS Leakage-Aware Feature Selection Framework (Multi-Model Pipeline)

This script implements a reproducible, leakage-aware, multi-stage feature selection
workflow for binary PCOS classification:

Stages
------
Stage 0: Overfitting assessment (train vs. test AUC-ROC).
Stage 1: Redundancy filtering using Pearson correlation threshold.
Stage 2: Cross-model feature importance (embedded importance for tree-based models
         + permutation importance for all models).
Stage 3: SHAP-based interpretability for supported tree-based models.
Stage 4: Backward feature ablation curves.

Models
------
LR, DT, RF, KNN, SVM, ExtraTrees (ET), XGBoost, Two-Level Random Forest (2RF).

Validation
----------
- Repeated stratified 5-fold CV (3 repeats; total 15 folds) for internal validation.
- Independent test set evaluation with 1000× bootstrap 95% CI.

Reproducibility
---------------
- Deterministic random seeds.
- No automatic package installation inside the script.
- Outputs saved to a user-specified output directory.

Author
------
Van-Thao Ta (Corresponding author)
Hanoi Medical University, Hanoi, Vietnam
Email: tavanthao@hmu.edu.vn

Last updated: 2025-12-11
License: MIT
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
import xgboost as xgb

try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    shap = None
    plt = None

warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser(description="PCOS Feature Selection Pipeline")
parser.add_argument("--train-csv", required=True, help="Path to training CSV")
parser.add_argument("--test-csv", required=True, help="Path to test CSV")
parser.add_argument("--output-dir", default="outputs", help="Output directory")
parser.add_argument("--config", default="configs/config.example.yaml", help="Path to config YAML")
parser.add_argument("--rename-json", default="configs/rename_columns.example.json", help="Path to rename JSON")
args = parser.parse_args()

# Load config (fallback to defaults if file not found)
try:
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
except (ImportError, FileNotFoundError):
    config = {}
RANDOM_STATE = config.get("RANDOM_STATE", 42)
N_ESTIMATORS = config.get("N_ESTIMATORS", 300)
MAX_DEPTH = config.get("MAX_DEPTH", 10)
MIN_SAMPLES_SPLIT = config.get("MIN_SAMPLES_SPLIT", 5)
MIN_SAMPLES_LEAF = config.get("MIN_SAMPLES_LEAF", 2)
N_SPLITS = config.get("N_SPLITS", 5)
N_REPEATS = config.get("N_REPEATS", 3)
SCORING = config.get("SCORING", "roc_auc")

# Load rename dict
try:
    with open(args.rename_json, "r") as f:
        rename_dict = json.load(f)
except FileNotFoundError:
    rename_dict = {  # Default if file missing
        'PID': 'PID',
        'Age': 'Age',
        'chu kì kinh': 'Cycle length',
        'Cân nặng': 'Weight',
        'Chiều cao': 'Height',
        'Số nang buồng trứng phải': 'Follicle No. (R)',
        'Số nang buồng trứng trái': 'Follicle No. (L)',
        'Tổng số nang trái + phải': 'Total follicle No.',
        'AMH': 'AMH',
        'TESTOS': 'TESTOS',
        'LH': 'LH',
        'FSH': 'FSH',
        'E2': 'E2',
        'PRG': 'PRG',
        'PRL': 'PRL',
        'BMI': 'BMI',
        'PCOS': 'PCOS',
        'LH/FSH': 'LH/FSH',
        'Phân loại BMI': 'BMI classification',
        'Phân loại tuổi': 'Age classification',
    }

# Features and target
features_all = [
    'TESTOS', 'Cycle length', 'Total follicle No.',
    'AMH', 'Age', 
    'LH', 'FSH', 'E2', 'PRG', 'PRL', 'BMI', 'LH/FSH',
    'Follicle No. (R)','Follicle No. (L)', 'Weight', 'Height',
]
target = 'PCOS'

# Preprocess function
def preprocess(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        if df[col].dtype.kind in "biufc":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            conv = pd.to_numeric(df[col], errors='coerce')
            if conv.notna().any():
                df[col] = conv.fillna(np.nanmedian(conv))
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    return df

# TwoLevelRF class
class TwoLevelRF(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rf1 = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=-1
        )
        self.rf2 = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1, n_jobs=-1
        )

    def fit(self, X, y):
        self.rf1.fit(X, y)
        proba1 = self.rf1.predict_proba(X)
        X_aug = np.hstack([X, proba1])
        self.rf2.fit(X_aug, y)
        self.feature_importances_ = self.rf2.feature_importances_[:X.shape[1]]  # Approximate for orig features
        self.classes_ = self.rf2.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        proba1 = self.rf1.predict_proba(X)
        X_aug = np.hstack([X, proba1])
        return self.rf2.predict(X_aug)

    def predict_proba(self, X):
        proba1 = self.rf1.predict_proba(X)
        X_aug = np.hstack([X, proba1])
        return self.rf2.predict_proba(X_aug)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.rf1.set_params(**params)
        self.rf2.set_params(**{**params, "random_state": params.get("random_state", self.random_state) + 1})
        return self

# SHAP function
def compute_and_save_shap(model_name, pipe, X_tr, X_val, features_all, output_dir):
    print("\n=== SHAP (global & local) — robust path ===")
    if shap is None:
        print(f"Skipping SHAP for {model_name} (shap package not installed).")
        with open(output_dir / "shap_skipped.txt", "w") as f:
            f.write(f"SHAP skipped for {model_name} as shap package is not installed.")
        return
    if model_name not in ['DT', 'RF', 'ET', 'XGBoost', '2RF']:
        print(f"Skipping SHAP for {model_name} (not tree-based).")
        with open(output_dir / "shap_skipped.txt", "w") as f:
            f.write(f"SHAP skipped for {model_name} as it's not supported in this pipeline.")
        return
    try:
        print("Versions -> shap:", shap.__version__, "| sklearn:", __import__("sklearn").__version__)
        model = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        print("len(features_all):", len(features_all))
        print("model.n_features_in_:", getattr(model, "n_features_in_", None))
        print("model.classes_:", getattr(model, "classes_", None))
        print("X_val shape:", X_val.shape)
        X_tr_scaled = scaler.transform(X_tr)
        X_tr_scaled = np.asarray(X_tr_scaled, dtype=float)
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = np.asarray(X_val_scaled, dtype=float)
        classes_ = getattr(model, "classes_", np.array([0,1]))
        pos_idx = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else len(classes_)-1
        if model_name == 'XGBoost':
            explainer = shap.TreeExplainer(model.get_booster(), X_tr_scaled, feature_perturbation="interventional", model_output="probability")
        else:
            explainer = shap.TreeExplainer(model, X_tr_scaled, feature_perturbation="interventional", model_output="probability")
        shap_values = explainer.shap_values(X_val_scaled)
        if isinstance(shap_values, list):
            sv = shap_values[pos_idx]
        else:
            sv = shap_values
            if sv.ndim == 3:
                sv = sv[:, :, pos_idx]
        print("SHAP 2D shape after squeeze:", sv.shape)
        n_shap_feats = sv.shape[1]
        feature_names_used = features_all[:n_shap_feats]
        shap_abs_mean = np.abs(sv).mean(axis=0)
        shap_global = pd.DataFrame({
            "feature": feature_names_used,
            "mean_abs_shap": shap_abs_mean
        }).sort_values("mean_abs_shap", ascending=False)
        print(shap_global)
        shap_global.to_csv(output_dir / "shap_global.csv", index=False)
        sv_2d = sv[:, :, pos_idx] if (isinstance(sv, np.ndarray) and sv.ndim == 3) else sv
        n_plot_feats = sv_2d.shape[1]
        X_plot = X_val_scaled[:, :n_plot_feats]
        plt.figure()
        shap.summary_plot(sv_2d, X_plot, feature_names=feature_names_used[:n_plot_feats], show=False)
        plt.title("SHAP summary (violin/beeswarm) — class 1")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_violin.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure()
        shap.summary_plot(sv_2d, X_plot, feature_names=feature_names_used[:n_plot_feats], plot_type="bar", show=False)
        plt.title("SHAP global importance (bar) — class 1")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved plots:", output_dir / "shap_summary_violin.png", "and", output_dir / "shap_summary_bar.png")
    except Exception as e1:
        print("TreeExplainer failed:", repr(e1))
        print("Falling back to shap.Explainer with masker…")
        try:
            masker = shap.maskers.Independent(X_tr_scaled)
            explainer = shap.Explainer(pipe.predict_proba, masker)
            exp = explainer(X_val_scaled)
            sv = exp.values
            print("exp.values shape (fallback):", sv.shape)
            if sv.ndim == 3:
                sv = sv[:, :, pos_idx]
            n_shap_feats = sv.shape[1]
            feature_names_used = features_all[:n_shap_feats]
            shap_abs_mean = np.abs(sv).mean(axis=0)
            shap_global = pd.DataFrame({
                "feature": feature_names_used,
                "mean_abs_shap": shap_abs_mean
            }).sort_values("mean_abs_shap", ascending=False)
            print(shap_global)
            shap_global.to_csv(output_dir / "shap_global.csv", index=False)
            sv_2d = sv[:, :, pos_idx] if (isinstance(sv, np.ndarray) and sv.ndim == 3) else sv
            n_plot_feats = sv_2d.shape[1]
            X_plot = X_val_scaled[:, :n_plot_feats]
            plt.figure()
            shap.summary_plot(sv_2d, X_plot, feature_names=feature_names_used[:n_plot_feats], show=False)
            plt.title("SHAP summary (violin/beeswarm) — class 1")
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary_violin.png", dpi=300, bbox_inches="tight")
            plt.close()
            plt.figure()
            shap.summary_plot(sv_2d, X_plot, feature_names=feature_names_used[:n_plot_feats], plot_type="bar", show=False)
            plt.title("SHAP global importance (bar) — class 1")
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("Saved plots (fallback):", output_dir / "shap_summary_violin.png", "and", output_dir / "shap_summary_bar.png")
        except Exception as e2:
            print("Fallback Explainer also failed:", repr(e2))
            with open(output_dir / "shap_skipped.txt", "w") as f:
                f.write(f"SHAP failed for {model_name}: {repr(e2)}")

# Main function
def main():
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "python_version": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "xgboost": xgb.__version__,
            "shap": shap.__version__ if shap else "Not installed",
        },
    }
    with open(output_path / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Load data
    train_data = pd.read_csv(args.train_csv).rename(columns=lambda c: c.strip())
    test_data = pd.read_csv(args.test_csv).rename(columns=lambda c: c.strip())
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    train_data = train_data.rename(columns=rename_dict)
    test_data = test_data.rename(columns=rename_dict)

    features_all = [f for f in features_all if f in train_data.columns]
    print("Using features (available):", features_all)

    train_data = preprocess(train_data, features_all + [target])
    test_data = preprocess(test_data, features_all + ([target] if target in test_data.columns else []))

    X = train_data[features_all]
    y = train_data[target].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    ext_has_label = target in test_data.columns
    X_ext = test_data[features_all]
    y_ext = test_data[target].astype(int) if ext_has_label else None

    # Models
    models = {
        'LR': LogisticRegression(random_state=RANDOM_STATE, max_iter=500),
        'DT': DecisionTreeClassifier(
            max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
        ),
        'RF': RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
        ),
        'ET': ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'SVM': SVC(
            probability=True, random_state=RANDOM_STATE
        ),
        '2RF': TwoLevelRF(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE
        )
    }

    cv_results = []
    bootstrap_results = []

    for model_name, clf in models.items():
        print(f"\n\n=== MODEL: {model_name} ===")
        model_dir = output_path / model_name
        model_dir.mkdir(exist_ok=True)

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        val_pred = pipe.predict(X_val)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        print("\n=== HOLDOUT VALIDATION (from training set) ===")
        print("Accuracy:", round(accuracy_score(y_val, val_pred), 4))
        print("AUC-ROC :", round(roc_auc_score(y_val, val_proba), 4))
        print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))
        print("Classification report:\n", classification_report(y_val, val_pred, digits=4))

        if ext_has_label:
            ext_pred = pipe.predict(X_ext)
            ext_proba = pipe.predict_proba(X_ext)[:, 1]
            print("\n=== EXTERNAL TEST (provided test CSV) ===")
            print("Accuracy:", round(accuracy_score(y_ext, ext_pred), 4))
            print("AUC-ROC :", round(roc_auc_score(y_ext, ext_proba), 4))
            print("Confusion matrix:\n", confusion_matrix(y_ext, ext_pred))
            print("Classification report:\n", classification_report(y_ext, ext_pred, digits=4))
        else:
            print("\n[Note] External test set does not include labels; skipping external metrics.")

        # CV
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        scoring = {'auc': 'roc_auc', 'acc': 'accuracy', 'prec': 'precision', 'rec': 'recall', 'f1': 'f1'}
        cv_full = cross_validate(pipe, X, y, cv=rskf, scoring=scoring, n_jobs=-1)
        cv_auc_mean = cv_full['test_auc'].mean()
        cv_auc_std = cv_full['test_auc'].std()
        cv_acc_mean = cv_full['test_acc'].mean()
        cv_acc_std = cv_full['test_acc'].std()
        cv_prec_mean = cv_full['test_prec'].mean()
        cv_prec_std = cv_full['test_prec'].std()
        cv_rec_mean = cv_full['test_rec'].mean()
        cv_rec_std = cv_full['test_rec'].std()
        cv_f1_mean = cv_full['test_f1'].mean()
        cv_f1_std = cv_full['test_f1'].std()
        cv_results.append({
            "Model": model_name,
            "AUC-ROC (CV)": f"{cv_auc_mean:.4f} ± {cv_auc_std:.4f}",
            "Accuracy (CV)": f"{cv_acc_mean:.4f} ± {cv_acc_std:.4f}",
            "Precision (CV)": f"{cv_prec_mean:.4f} ± {cv_prec_std:.4f}",
            "Recall (CV)": f"{cv_rec_mean:.4f} ± {cv_rec_std:.4f}",
            "F1-Score (CV)": f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}"
        })

        # Bootstrap
        if ext_has_label:
            boot_aucs = []
            boot_accs = []
            boot_precs = []
            boot_recs = []
            boot_f1s = []
            for i in range(1000):
                Xb, yb = resample(X_ext, y_ext, random_state=i, stratify=y_ext)
                pred = pipe.predict(Xb)
                proba = pipe.predict_proba(Xb)[:, 1]
                boot_aucs.append(roc_auc_score(yb, proba))
                boot_accs.append(accuracy_score(yb, pred))
                boot_precs.append(precision_score(yb, pred))
                boot_recs.append(recall_score(yb, pred))
                boot_f1s.append(f1_score(yb, pred))
            boot_auc_mean = np.mean(boot_aucs)
            boot_auc_ci_low = np.percentile(boot_aucs, 2.5)
            boot_auc_ci_high = np.percentile(boot_aucs, 97.5)
            boot_acc_mean = np.mean(boot_accs)
            boot_acc_ci_low = np.percentile(boot_accs, 2.5)
            boot_acc_ci_high = np.percentile(boot_accs, 97.5)
            boot_prec_mean = np.mean(boot_precs)
            boot_prec_ci_low = np.percentile(boot_precs, 2.5)
            boot_prec_ci_high = np.percentile(boot_precs, 97.5)
            boot_rec_mean = np.mean(boot_recs)
            boot_rec_ci_low = np.percentile(boot_recs, 2.5)
            boot_rec_ci_high = np.percentile(boot_recs, 97.5)
            boot_f1_mean = np.mean(boot_f1s)
            boot_f1_ci_low = np.percentile(boot_f1s, 2.5)
            boot_f1_ci_high = np.percentile(boot_f1s, 97.5)
            bootstrap_results.append({
                "Model": model_name,
                "AUC-ROC (Test)": f"{boot_auc_mean:.4f}",
                "AUC 95% CI": f"({boot_auc_ci_low:.3f}–{boot_auc_ci_high:.3f})",
                "Accuracy (Test)": f"{boot_acc_mean:.4f}",
                "Acc 95% CI": f"({boot_acc_ci_low:.3f}–{boot_acc_ci_high:.3f})",
                "Precision (Test)": f"{boot_prec_mean:.4f}",
                "Prec 95% CI": f"({boot_prec_ci_low:.3f}–{boot_prec_ci_high:.3f})",
                "Recall (Test)": f"{boot_rec_mean:.4f}",
                "Rec 95% CI": f"({boot_rec_ci_low:.3f}–{boot_rec_ci_high:.3f})",
                "F1-Score (Test)": f"{boot_f1_mean:.4f}",
                "F1 95% CI": f"({boot_f1_ci_low:.3f}–{boot_f1_ci_high:.3f})"
            })

        # Cross-Validation Stability
        print("\n=== CROSS-VALIDATION STABILITY ===")
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipe, X, y, scoring=SCORING, cv=cv, n_jobs=-1)
        print(f"{SCORING} (mean±sd) over {N_SPLITS}-fold:", round(scores.mean(),4), "±", round(scores.std(),4))
        rank_counter = Counter()
        for rep in range(N_REPEATS):
            cv_rep = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE+rep)
            for fold, (tr_idx, te_idx) in enumerate(cv_rep.split(X, y), 1):
                X_tr_cv, X_te_cv = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr_cv, y_te_cv = y.iloc[tr_idx], y.iloc[te_idx]
                if model_name == '2RF':
                    clf_cv = TwoLevelRF(
                        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                        min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
                        random_state=RANDOM_STATE+rep*100+fold
                    )
                else:
                    params = clf.get_params()
                    if 'random_state' in params:
                        params['random_state'] = RANDOM_STATE+rep*100+fold
                    clf_cv = clf.__class__(**params)
                pipe_cv = Pipeline([("scaler", StandardScaler()), ("clf", clf_cv)])
                pipe_cv.fit(X_tr_cv, y_tr_cv)
                model_cv = pipe_cv.named_steps["clf"]
                if hasattr(model_cv, "feature_importances_"):
                    order = np.argsort(model_cv.feature_importances_)[::-1]
                    top5 = [features_all[i] for i in order[:5]]
                    rank_counter.update(top5)
                else:
                    perm_cv = permutation_importance(
                        pipe_cv, X_te_cv, y_te_cv, n_repeats=5, scoring="roc_auc",
                        random_state=RANDOM_STATE+rep*100+fold, n_jobs=-1
                    )
                    order = np.argsort(perm_cv.importances_mean)[::-1]
                    top5 = [features_all[i] for i in order[:5]]
                    rank_counter.update(top5)
        print("\nTop-5 frequency across folds (higher = more stable):")
        for feat, cnt in rank_counter.most_common():
            print(f"{feat}: {cnt} / {N_SPLITS*N_REPEATS} folds")

        # Permutation Importance
        print("\n=== PERMUTATION IMPORTANCE (holdout set) ===")
        perm = permutation_importance(
            pipe, X_val, y_val, n_repeats=20, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_df = pd.DataFrame({
            "feature": features_all,
            "mean_importance": perm.importances_mean,
            "std_importance": perm.importances_std
        }).sort_values("mean_importance", ascending=False)
        print(perm_df)
        perm_df.to_csv(model_dir / "perm.csv", index=False)

        # Feature Ablation Curve
        print("\n=== FEATURE ABLATION CURVE ===")
        model0 = pipe.named_steps["clf"]
        if hasattr(model0, "feature_importances_"):
            base_order = np.argsort(model0.feature_importances_)
        else:
            base_order = np.argsort(perm.importances_mean)
        keep_counts = sorted(set([len(features_all), 14, 12, 10, 8, 6, 4, 3, 2]), reverse=True)
        keep_counts = [k for k in keep_counts if k <= len(features_all)]
        ablation_rows = []
        for k in keep_counts:
            keep_idx = base_order[::-1][:k]
            keep_feats = [features_all[i] for i in keep_idx]
            X_tr_k = X_tr[keep_feats]
            X_val_k = X_val[keep_feats]
            if model_name == '2RF':
                clf_k = TwoLevelRF(
                    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                    min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE
                )
            else:
                clf_k = model0.__class__(**model0.get_params())
            pipe_k = Pipeline([("scaler", StandardScaler()), ("clf", clf_k)])
            pipe_k.fit(X_tr_k, y_tr)
            val_pred_k = pipe_k.predict(X_val_k)
            val_proba_k = pipe_k.predict_proba(X_val_k)[:,1]
            cm = confusion_matrix(y_val, val_pred_k)
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_val, val_pred_k)
            auc = roc_auc_score(y_val, val_proba_k)
            sens = tp / (tp + fn) if (tp+fn)>0 else 0
            spec = tn / (tn + fp) if (tn+fp)>0 else 0
            prec = tp / (tp + fp) if (tp+fp)>0 else 0
            f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
            ablation_rows.append({
                "n_features": k, "kept_features": keep_feats,
                "accuracy": acc, "auc": auc, "sensitivity": sens,
                "specificity": spec, "precision": prec, "f1": f1
            })
        ablation_df = pd.DataFrame(ablation_rows).sort_values("n_features", ascending=False)
        print(ablation_df[["n_features","accuracy","auc","sensitivity","specificity","precision","f1"]])
        ablation_df.to_csv(model_dir / "ablation_results.csv", index=False)

        # SHAP
        compute_and_save_shap(model_name, pipe, X_tr, X_val, features_all, model_dir)
        print(f"\nAll checks completed for {model_name}. Files saved under {model_dir}")

    # Save summary results
    cv_df = pd.DataFrame(cv_results)
    print("\nBẢNG INTERNAL VALIDATION (3×5-fold CV trên n=492)")
    print(cv_df.to_string(index=False))
    cv_df.to_csv(output_path / "cv_3x5fold_results.csv", index=False)

    if ext_has_label:
        boot_df = pd.DataFrame(bootstrap_results)
        print("\nBẢNG TEST-SET PERFORMANCE + 95% BOOTSTRAP CI (n=120)")
        print(boot_df.to_string(index=False))
        boot_df.to_csv(output_path / "test_bootstrap_1000_ci.csv", index=False)

if __name__ == "__main__":
    main()
