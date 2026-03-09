"""
Module 6: evaluation.py
------------------------
Compute metrics, confidence intervals, statistical tests, and generate
comparison figures for the report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, and F1 (macro) as a dict."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1_macro':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def compute_ci(scores, confidence=0.95):
    """
    Compute confidence interval for a list of CV fold scores.

    Uses t-distribution (appropriate for small n like 5 folds).

    Returns
    -------
    (lower, upper) tuple
    """
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)   # standard error of the mean
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return (mean - h, mean + h)


def paired_ttest(scores_a, scores_b):
    """
    Paired t-test between two sets of CV scores.

    Returns
    -------
    t_stat  : t-statistic
    p_value : two-tailed p-value
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def plot_confusion_matrix(y_true, y_pred, labels, title='', save_path=None):
    """Confusion matrix as a seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curve(model, X, y, save_path=None):
    """
    One-vs-rest ROC curve for each class.
    Requires model with predict_proba support.
    """
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    y_prob = model.predict_proba(X)

    plt.figure(figsize=(7, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Speaker {cls} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.title('ROC Curve (One-vs-Rest)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison_table(results_path='results.json', save_path=None):
    """
    Bar chart comparing Pipeline A vs Pipeline B accuracy.
    Reads from results.json.
    """
    with open(results_path) as f:
        results = json.load(f)

    experiments = results['experiments']
    names  = list(experiments.keys())
    means  = [experiments[n]['accuracy']['mean'] for n in names]
    ci_low = [experiments[n]['accuracy']['ci_95'][0] for n in names]
    ci_up  = [experiments[n]['accuracy']['ci_95'][1] for n in names]
    errors = [m - lo for m, lo in zip(means, ci_low)]

    colors = ['steelblue', 'darkorange']
    x = np.arange(len(names))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, means, yerr=errors, capsize=6,
                   color=colors[:len(names)], alpha=0.85)
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel('Mean CV Accuracy')
    plt.title('Pipeline Comparison — Accuracy with 95% CI')
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, m + 0.02,
                 f'{m:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
