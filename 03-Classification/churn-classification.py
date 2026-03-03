#!/usr/bin/env python
# coding: utf-8

"""
Telco Customer Churn Prediction (Logistic Regression)
Clean script version + no warnings (pandas dtype + convergence)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/"
    "chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]

NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]


def load_and_prepare(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)

    # Normalize text values safely for both object + pandas string dtype
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip().str.lower().str.replace(" ", "_", regex=False)

    # totalcharges to numeric + fill missing
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # churn to binary (after lowercasing => "yes")
    df["churn"] = (df["churn"] == "yes").astype(int)

    # Safety check
    if df["churn"].nunique() < 2:
        raise ValueError(f"Target has only one class:\n{df['churn'].value_counts()}")

    return df


def split(df: pd.DataFrame, seed: int = 1):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train["churn"].values
    y_val = df_val["churn"].values
    y_test = df_test["churn"].values

    X_train = df_train.drop(columns=["churn"])
    X_val = df_val.drop(columns=["churn"])
    X_test = df_test.drop(columns=["churn"])

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    # Preprocess: one-hot encode categoricals, scale numericals
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ]
    )

    # Model: Logistic Regression
    clf = LogisticRegression(solver="lbfgs", max_iter=3000)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def accuracy_at_threshold(model, X, y, threshold: float = 0.5) -> float:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return accuracy_score(y, pred)


def main():
    df = load_and_prepare(DATA_URL)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df, seed=1)

    model = build_model()
    model.fit(X_train[CATEGORICAL + NUMERICAL], y_train)

    val_acc = accuracy_at_threshold(model, X_val[CATEGORICAL + NUMERICAL], y_val, threshold=0.5)
    test_acc = accuracy_at_threshold(model, X_test[CATEGORICAL + NUMERICAL], y_test, threshold=0.5)

    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Test accuracy:       {test_acc:.3f}")

    # Example inference on last test customer
    customer = X_test.iloc[-1][CATEGORICAL + NUMERICAL].to_frame().T
    churn_proba = model.predict_proba(customer)[0, 1]
    print(f"\nExample customer churn probability: {churn_proba:.3f}")


if __name__ == "__main__":
    main()