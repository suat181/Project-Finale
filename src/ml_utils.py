"""
Shared modeling utilities for the Video-Game Hit/Flop project — Phase C.

EVERY teammate imports from this module so all results are directly comparable
(same metrics, same encoding rules, same output format).

- Encoding rules follow DECISIONS [M-08]:
    * Naive Bayes (CategoricalNB) -> ORDINAL (integer) encoding
    * everyone else (kNN, Decision Tree, Random Forest, LogReg, MLP) -> ONE-HOT
- Metrics follow COURSE.md (Prof. Goldstein's toolbox): confusion matrix,
  precision / recall / F1, ROC-AUC, plus the null-accuracy baseline.
- Leakage rules (hard rules): encoders/scalers live INSIDE the Pipeline so they
  are fit on the training fold only; downsampling is applied to TRAIN ONLY.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc,
)

RANDOM_STATE = 42
CAT_COLS = ["Genre_Clean", "Platform_Family", "Publisher_Tier"]


def load_scope(scope_path):
    """Load X_train/X_test/y_train/y_test from a data/processed/<scope> folder."""
    X_train = pd.read_csv(f"{scope_path}/X_train.csv")
    X_test = pd.read_csv(f"{scope_path}/X_test.csv")
    y_train = pd.read_csv(f"{scope_path}/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(f"{scope_path}/y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def make_encoder(kind, cat_cols=CAT_COLS):
    """Return a ColumnTransformer for the categorical features (DECISIONS [M-08]).

    kind="onehot"  -> kNN / DecisionTree / RandomForest / LogReg / MLP
    kind="ordinal" -> CategoricalNB (needs integer-coded categories, NOT one-hot)
    """
    if kind == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    elif kind == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore")
    else:
        raise ValueError("kind must be 'onehot' or 'ordinal'")
    return ColumnTransformer([("cat", enc, cat_cols)])


def downsample_5050(X, y, random_state=RANDOM_STATE):
    """Manual random undersampling to a 50/50 class balance — TRAIN ONLY.

    This is Prof. Goldstein's 'balance the classes during training' (Lec 5, slide 34).
    NEVER call this on the test set.
    """
    y = pd.Series(np.asarray(y))
    n = y.value_counts().min()
    keep = []
    for cls in y.unique():
        keep.extend(y[y == cls].sample(n=n, random_state=random_state).index)
    Xb = X.reset_index(drop=True).loc[keep].reset_index(drop=True)
    yb = y.loc[keep].reset_index(drop=True)
    return Xb, yb


def null_accuracy(y):
    """Baseline: a classifier that always predicts the majority class.
    Our models must beat this to be useful (COURSE.md / kNN-Evaluation notebook)."""
    y = np.asarray(y)
    return max((y == 0).mean(), (y == 1).mean())


def evaluate(name, pipe, X_train, y_train, X_test, y_test,
             results_csv=None, plot=True):
    """Fit on train, evaluate on test with Prof. Goldstein's metric toolbox.

    Returns a one-row dict; optionally appends it to a shared results CSV so the
    three teammates' numbers land in one comparable table.
    """
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # probability/score for ROC-AUC (fall back gracefully)
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        y_score = pipe.decision_function(X_test)
    else:
        y_score = y_pred

    row = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "null_acc": round(null_accuracy(y_test), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print(f"ROC-AUC: {row['roc_auc']} | F1: {row['f1']} | "
          f"null-accuracy baseline: {row['null_acc']}")

    if plot:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.show()

    if results_csv:
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
        pd.DataFrame([row]).to_csv(
            results_csv, mode="a", header=not os.path.exists(results_csv), index=False
        )
    return row


def plot_roc_curves(fitted_models, X_test, y_test, title="ROC comparison"):
    """Overlay ROC curves for several already-fitted pipelines (for the C2 comparison).
    `fitted_models` = {name: fitted_pipeline}. Perfect = top-left, diagonal = random."""
    plt.figure(figsize=(7, 6))
    for name, pipe in fitted_models.items():
        if hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(X_test)[:, 1]
        else:
            score = pipe.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.show()
