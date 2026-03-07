#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Churn classification (Logistic Regression) with:
- Clean preprocessing
- Train/Validation/Test split (stratified)
- DictVectorizer one-hot encoding
- Accuracy + ROC-AUC + ROC curve
- Optional 5-fold CV ROC-AUC

This version avoids the common bug of overwriting y_val/y_pred during CV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import KFold, train_test_split


# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "data-week-3.csv"
RANDOM_STATE = 1

NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]
CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]
FEATURES = CATEGORICAL + NUMERICAL


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # normalize string columns (including churn)
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    # numeric conversion
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # robust churn conversion
    df["churn"] = (df["churn"] == "yes").astype(int)

    return df


# -----------------------------
# Splitting
# -----------------------------
def make_splits(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["churn"],
    )

    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,  # 0.25 of 0.8 = 0.2 → 60/20/20 split
        random_state=RANDOM_STATE,
        stratify=df_full_train["churn"],
    )

    # reset indices (nice for debugging)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test, df_full_train


# -----------------------------
# Vectorization (one-hot)
# -----------------------------
def fit_vectorizer(df_train: pd.DataFrame) -> DictVectorizer:
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train[FEATURES].to_dict(orient="records")
    dv.fit(train_dicts)
    return dv


def transform(dv: DictVectorizer, df_part: pd.DataFrame) -> np.ndarray:
    dicts = df_part[FEATURES].to_dict(orient="records")
    return dv.transform(dicts)


# -----------------------------
# Model
# -----------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]

    return acc, auc, cm


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Model (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend()
    plt.show()


# -----------------------------
# Cross-validation (optional)
# -----------------------------
def cv_auc(df_full_train: pd.DataFrame, C: float = 1.0, n_splits: int = 5) -> tuple[float, float]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx].reset_index(drop=True)
        df_va = df_full_train.iloc[val_idx].reset_index(drop=True)

        y_tr = df_tr["churn"].values
        y_va = df_va["churn"].values

        dv = fit_vectorizer(df_tr)
        X_tr = transform(dv, df_tr)
        X_va = transform(dv, df_va)

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_tr, y_tr)

        y_score = model.predict_proba(X_va)[:, 1]
        scores.append(roc_auc_score(y_va, y_score))

    return float(np.mean(scores)), float(np.std(scores))


# -----------------------------
# Main
# -----------------------------
def main():
    df = load_and_clean(CSV_PATH)

    print("Churn distribution:")
    print(df["churn"].value_counts(), "\n")

    df_train, df_val, df_test, df_full_train = make_splits(df)

    # Prepare target arrays
    y_train = df_train["churn"].values
    y_val = df_val["churn"].values
    y_test = df_test["churn"].values

    # Vectorize
    dv = fit_vectorizer(df_train)
    X_train = transform(dv, df_train)
    X_val = transform(dv, df_val)
    X_test = transform(dv, df_test)

    # Train
    model = train_model(X_train, y_train)

    # Validate
    y_val_score = model.predict_proba(X_val)[:, 1]
    acc_val, auc_val, cm_val = evaluate_binary(y_val, y_val_score)

    print(f"Validation accuracy: {acc_val:.3f}")
    print(f"Validation AUC:      {auc_val:.3f}")
    print("Confusion matrix (val) [[TN, FP], [FN, TP]]:")
    print(cm_val, "\n")

    plot_roc(y_val, y_val_score, "ROC Curve (Validation)")

    # Test
    y_test_score = model.predict_proba(X_test)[:, 1]
    acc_test, auc_test, cm_test = evaluate_binary(y_test, y_test_score)

    print(f"Test accuracy: {acc_test:.3f}")
    print(f"Test AUC:      {auc_test:.3f}")
    print("Confusion matrix (test) [[TN, FP], [FN, TP]]:")
    print(cm_test, "\n")

    # Optional cross-validation
    mean_auc, std_auc = cv_auc(df_full_train, C=1.0, n_splits=5)
    print(f"5-Fold CV AUC: {mean_auc:.3f} ± {std_auc:.3f}")


if __name__ == "__main__":
    main()