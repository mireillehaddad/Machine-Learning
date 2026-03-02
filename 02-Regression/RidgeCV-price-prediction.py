#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # -------------------------
    # Load + clean
    # -------------------------
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    # -------------------------
    # Target + feature engineering
    # -------------------------
    y = np.log1p(df["msrp"].values)
    df["age"] = 2017 - df["year"]
    X = df.drop(columns=["msrp"])

    # -------------------------
    # Split 60/20/20
    # -------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=2
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=2
    )

    # -------------------------
    # Features
    # -------------------------
    numeric_features = [
        "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg",
        "popularity", "age", "number_of_doors", "year",
    ]

    categorical_features = [
        "make", "model", "engine_fuel_type", "driven_wheels",
        "market_category", "vehicle_size", "vehicle_style",
    ]

    # -------------------------
    # Preprocess
    # -------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=0), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop"
    )

    # -------------------------
    # Ridge + alpha search
    # -------------------------
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge())
    ])

    alphas = np.logspace(-6, 2, 20)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"model__alpha": alphas},
        scoring="neg_root_mean_squared_error",  # RMSE on log(msrp)
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    print("Chosen alpha:", grid.best_params_["model__alpha"])

    # -------------------------
    # Evaluate
    # -------------------------
    val_pred = best_pipe.predict(X_val)
    test_pred = best_pipe.predict(X_test)

    print("RMSE (validation, log):", rmse(y_val, val_pred))
    print("RMSE (test, log):", rmse(y_test, test_pred))

    print("RMSE (validation, $):", rmse(np.expm1(y_val), np.expm1(val_pred)))

    # -------------------------
    # Predict one sample
    # -------------------------
    sample = X_test.iloc[[20]]
    pred_log = best_pipe.predict(sample)[0]
    pred_price = np.expm1(pred_log)

    print("Predicted MSRP for sample:", pred_price)


if __name__ == "__main__":
    main()