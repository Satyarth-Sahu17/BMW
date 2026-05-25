"""
Comprehensive evaluation metrics and confusion matrix computation.
This is the core evaluation module with all required metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef,
    log_loss, brier_score_loss,
    roc_curve, precision_recall_curve
)
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


def compute_confusion_matrix(y_true, y_pred, labels, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    return cm


def plot_confusion_matrix(
    cm,
    labels,
    normalize=False,
    title="Confusion Matrix",
    save_path=None,
    figsize=(10, 8)
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def compute_all_metrics(y_true, y_pred, y_pred_proba=None, labels=None):
    metrics = {}

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    for avg in ['micro', 'macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f'precision_{avg}'] = float(p)
        metrics[f'recall_{avg}'] = float(r)
        metrics[f'f1_score_{avg}'] = float(f)

    if labels is not None:
        metrics['per_class'] = {}
        for idx, label in enumerate(labels):
            metrics['per_class'][label] = {
                "precision": float(precision[idx]) if idx < len(precision) else 0.0,
                "recall": float(recall[idx]) if idx < len(recall) else 0.0,
                "f1_score": float(f1[idx]) if idx < len(f1) else 0.0,
                "support": int(support[idx]) if idx < len(support) else 0
            }

    metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))

    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))

            n_classes = y_pred_proba.shape[1]
            if n_classes == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:
                metrics['roc_auc_ovr_macro'] = float(
                    roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro")
                )

        except Exception as e:
            print(f"Warning: Probability metrics error: {e}")

    return metrics


def plot_roc_curves(y_true, y_pred_proba, labels, save_path=None, figsize=(10, 8)):
    from sklearn.preprocessing import label_binarize

    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        if y_true_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curves(y_true, y_pred_proba, labels, save_path=None, figsize=(10, 8)):
    from sklearn.preprocessing import label_binarize

    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        if y_true_bin[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")

    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_most_confused_pairs(cm, labels, top_n=5):
    pairs = []
    n = len(labels)

    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def generate_evaluation_report(
    y_true, y_pred, y_pred_proba, labels, output_dir, experiment_name="evaluation"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating evaluation report in {output_dir}...")

    # ---- METRICS JSON (UTF-8 FIXED)
    metrics = compute_all_metrics(y_true, y_pred, y_pred_proba, labels)
    metrics_file = output_path / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("✓ Metrics saved")

    # ---- CONFUSION MATRIX
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    plot_confusion_matrix(cm, labels, save_path=str(output_path / "confusion_matrix_absolute.png"))
    print("✓ Confusion matrix saved")

    cm_norm = compute_confusion_matrix(y_true, y_pred, labels, normalize=True)
    plot_confusion_matrix(cm_norm, labels, normalize=True,
                          save_path=str(output_path / "confusion_matrix_normalized.png"))

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_path / "confusion_matrix.csv")

    confused_pairs = find_most_confused_pairs(cm, labels)

    # ---- ROC / PR curves
    if y_pred_proba is not None:
        plot_roc_curves(y_true, y_pred_proba, labels, save_path=str(output_path / "roc_curves.png"))
        plot_precision_recall_curves(y_true, y_pred_proba, labels,
                                     save_path=str(output_path / "pr_curves.png"))
        print("✓ ROC & PR curves saved")

    # ---- HTML REPORT (UTF-8 FIXED)
    html = generate_html_report(metrics, cm, labels, confused_pairs, experiment_name)
    html_file = output_path / "report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ HTML report saved: {html_file}")

    print("\n✓ Evaluation report complete!")


def generate_html_report(metrics, cm, labels, confused_pairs, experiment_name):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>BMW Evaluation Report - {experiment_name}</title>
    </head>
    <body>
        <h1>🎯 Evaluation Report</h1>
        <p>Accuracy: {metrics['accuracy']:.4f}</p>
        <p>Balanced Accuracy: {metrics['balanced_accuracy']:.4f}</p>
        <p>F1 Macro: {metrics['f1_score_macro']:.4f}</p>
    </body>
    </html>
    """
    return html
