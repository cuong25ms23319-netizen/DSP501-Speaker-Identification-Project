"""
Module 5: train.py
------------------
Train and evaluate SVM models for both pipelines.

Experiments:
  A1 — SVM on basic time-domain features (Pipeline A — no DSP)
  B1 — SVM on MFCC after FIR + pre-emphasis (Pipeline B — with DSP)

Training strategy:
  - StratifiedKFold (5 folds) — preserves class balance
  - GridSearchCV for hyperparameter tuning inside each fold
  - StandardScaler inside a Pipeline (prevents data leakage)
  - Random seed: 42 everywhere
"""

import os
import json
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from evaluation import compute_ci, paired_ttest


RANDOM_SEED = 42
CV_FOLDS = 5

# SVM hyperparameter search space
SVM_PARAM_GRID = {
    'svm__C':     [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01],
}


def train_svm(X, y):
    """
    Train an SVM with GridSearchCV and 5-fold stratified CV.

    Returns
    -------
    best_model : fitted sklearn Pipeline (scaler + SVM)
    cv_scores  : accuracy on each of the 5 folds
    best_params: dict of best hyperparameters
    """
    # Build a pipeline so the scaler is fit only on training data per fold
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Inner CV for hyperparameter search (3-fold to keep it fast)
    grid_search = GridSearchCV(
        pipe, SVM_PARAM_GRID,
        cv=3, scoring='accuracy', n_jobs=-1, refit=True
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = {k.replace('svm__', ''): v
                   for k, v in grid_search.best_params_.items()}

    # Outer CV: evaluate the best model across 5 folds
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')

    return best_model, cv_scores, best_params


def run_experiment(name, X, y):
    """
    Run one full experiment: train, evaluate, collect metrics.

    Returns a dict with all results for results.json.
    """
    print(f"\n=== Experiment: {name} ===")
    model, cv_scores, best_params = train_svm(X, y)

    mean_acc = float(cv_scores.mean())
    std_acc = float(cv_scores.std())
    ci = compute_ci(cv_scores)

    # Compute additional metrics on the full dataset (for reporting)
    y_pred = model.predict(X)
    f1  = float(f1_score(y, y_pred, average='macro'))
    prec = float(precision_score(y, y_pred, average='macro', zero_division=0))
    rec  = float(recall_score(y, y_pred, average='macro', zero_division=0))

    print(f"  Best params : {best_params}")
    print(f"  CV Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  95% CI      : [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  F1 macro    : {f1:.4f}")

    return {
        'best_params': best_params,
        'cv_scores': cv_scores.tolist(),
        'accuracy':  {'mean': mean_acc, 'std': std_acc, 'ci_95': list(ci)},
        'f1_macro':  {'mean': f1},
        'precision': {'mean': prec},
        'recall':    {'mean': rec},
    }, model


def save_results(results, path='results.json'):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {path}")


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load pre-extracted features
    X_basic = np.load(os.path.join(ROOT, 'features', 'features_basic.npy'))
    X_mfcc  = np.load(os.path.join(ROOT, 'features', 'features_mfcc_filt.npy'))
    y_basic = np.load(os.path.join(ROOT, 'features', 'labels_basic.npy'))
    y_mfcc  = np.load(os.path.join(ROOT, 'features', 'labels_mfcc.npy'))

    # Run experiments
    res_a1, model_a1 = run_experiment('A1_SVM_basic', X_basic, y_basic)
    res_b1, model_b1 = run_experiment('B1_SVM_dsp',   X_mfcc,  y_mfcc)

    # Statistical test: does filtering improve SVM?
    t_stat, p_value = paired_ttest(
        res_a1['cv_scores'],
        res_b1['cv_scores']
    )
    print(f"\nPaired t-test (A1 vs B1): t={t_stat:.4f}, p={p_value:.4f}")

    # Save all results
    results = {
        'random_seed': RANDOM_SEED,
        'cv_folds': CV_FOLDS,
        'experiments': {
            'A1_SVM_basic': res_a1,
            'B1_SVM_dsp':   res_b1,
        },
        'statistical_tests': {
            'SVM_A_vs_B': {'t_stat': t_stat, 'p_value': p_value}
        }
    }
    save_results(results, path=os.path.join(ROOT, 'results.json'))

    # Save trained models
    models_dir = os.path.join(ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model_a1, os.path.join(models_dir, 'svm_pipeline_a.pkl'))
    joblib.dump(model_b1, os.path.join(models_dir, 'svm_pipeline_b.pkl'))
    print("Models saved → models/")


if __name__ == '__main__':
    main()
