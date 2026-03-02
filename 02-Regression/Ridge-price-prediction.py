#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------
# Utility
# ---------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Normalize string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    # Target (log scale)
    y = np.log1p(df["msrp"].values)

    # Feature engineering
    df["age"] = 2017 - df["year"]
    X = df.drop(columns=["msrp"])

    # -------------------------
    # Train / Val / Test split
    # -------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=2
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=2
    )

    # -------------------------
    # Feature groups
    # -------------------------
    numeric_features = [
        "engine_hp",
        "engine_cylinders",
        "highway_mpg",
        "city_mpg",
        "popularity",
        "age",
        "number_of_doors",
        "year"
    ]

    categorical_features = [
        "make",
        "model",
        "engine_fuel_type",
        "driven_wheels",
        "market_category",
        "vehicle_size",
        "vehicle_style"
    ]

    # -------------------------
    # Preprocessing
    # -------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # -------------------------
    # Model (Ridge = L2 regularization)
    # -------------------------
    model = Ridge(alpha=0.001)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    # -------------------------
    # Train
    # -------------------------
    pipe.fit(X_train, y_train)

    # -------------------------
    # Evaluate
    # -------------------------
    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    print("RMSE (validation, log):", rmse(y_val, val_pred))
    print("RMSE (test, log):", rmse(y_test, test_pred))

    print("RMSE (validation, $):",
          rmse(np.expm1(y_val), np.expm1(val_pred)))

    # -------------------------
    # Predict one example
    # -------------------------
    sample = X_test.iloc[20:21]
    pred_log = pipe.predict(sample)[0]
    pred_price = np.expm1(pred_log)

    print("Predicted MSRP for sample:", pred_price)


# ---------------------------------------------------

if __name__ == "__main__":
    main()