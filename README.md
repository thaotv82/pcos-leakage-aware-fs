#PCOS Leakage-Aware Feature Selection Framework

A multi-stage, leakage-aware feature selection workflow for binary PCOS classification using multiple machine learning models.

## Features
- Supports 8 classifiers: Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Extra Trees (ET), XGBoost, and Two-Level Random Forest (2RF)
- Repeated stratified 5-fold cross-validation (3 repeats, total 15 folds)
- Independent test set evaluation with 1000× bootstrap 95% confidence intervals
- Permutation importance (20 repeats)
- SHAP explanations for tree-based models (TreeExplainer with fallback)
- Backward feature ablation curves

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Usage
Bashpython src/pcos_pipeline.py \
  --train-csv data/pcos_training_492_new.csv \
  --test-csv data/pcos_test_120_new.csv \
  --output-dir outputs
Data Format
The script expects two CSV files:

Training set (~492 samples)
Test set (~120 samples)

Required columns:

Target: PCOS (0/1)
Features: As listed in configs/rename_columns.example.json

Important: Raw patient data are not included in this repository due to privacy concerns.
Notes

SHAP analysis is performed only for tree-based models (DT, RF, ET, XGBoost, 2RF).
If the shap package is not installed, SHAP steps will be skipped gracefully.
All random seeds are fixed for full reproducibility.
Outputs (CSV results, plots, metadata) are saved in the specified --output-dir.

License
MIT
