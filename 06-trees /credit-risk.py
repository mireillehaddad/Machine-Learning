#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# =========================
# 1. Load and clean data
# =========================

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv"
df = pd.read_csv(url)

df.columns = df.columns.str.lower()

status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}
df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}
df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)

for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(99999999, np.nan)


# =========================
# 2. Split data
# =========================

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = (df_full_train.status == 'default').astype(int).values
y_train = (df_train.status == 'default').astype(int).values
y_val = (df_val.status == 'default').astype(int).values
y_test = (df_test.status == 'default').astype(int).values

df_full_train = df_full_train.drop(columns=['status'])
df_train = df_train.drop(columns=['status'])
df_val = df_val.drop(columns=['status'])
df_test = df_test.drop(columns=['status'])


# =========================
# 3. Feature preparation
# =========================

categorical = ['job', 'marital', 'home', 'records']
numerical = ['seniority', 'time', 'age', 'expenses', 'income', 'assets', 'debt', 'amount', 'price']
features = categorical + numerical

df_train[numerical] = df_train[numerical].fillna(0)
df_val[numerical] = df_val[numerical].fillna(0)
df_full_train[numerical] = df_full_train[numerical].fillna(0)
df_test[numerical] = df_test[numerical].fillna(0)

dv = DictVectorizer(sparse=True)

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')
full_train_dicts = df_full_train[features].to_dict(orient='records')
test_dicts = df_test[features].to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# for final model, create a separate vectorizer fit on full training data
dv_full = DictVectorizer(sparse=True)
X_full_train = dv_full.fit_transform(full_train_dicts)
X_test = dv_full.transform(test_dicts)


# =========================
# 4. Decision Tree
# =========================

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=1)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict_proba(X_val)[:, 1]
auc_dt = roc_auc_score(y_val, y_pred_dt)


# =========================
# 5. Random Forest
# =========================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_val)[:, 1]
auc_rf = roc_auc_score(y_val, y_pred_rf)


# =========================
# 6. XGBoost on train/val
# =========================

feature_names = list(dv.get_feature_names_out())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 30,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

watchlist = [(dtrain, 'train'), (dval, 'val')]

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=175,
    evals=watchlist,
    verbose_eval=False
)

y_pred_xgb = xgb_model.predict(dval)
auc_xgb = roc_auc_score(y_val, y_pred_xgb)


# =========================
# 7. Compare models
# =========================

print("Validation AUC:")
print(f"Decision Tree : {auc_dt:.3f}")
print(f"Random Forest : {auc_rf:.3f}")
print(f"XGBoost       : {auc_xgb:.3f}")


# =========================
# 8. Final model on full train
# =========================

full_feature_names = list(dv_full.get_feature_names_out())

dfulltrain = xgb.DMatrix(
    X_full_train,
    label=y_full_train,
    feature_names=full_feature_names
)

dtest = xgb.DMatrix(
    X_test,
    label=y_test,
    feature_names=full_feature_names
)

final_model = xgb.train(
    params=xgb_params,
    dtrain=dfulltrain,
    num_boost_round=175,
    verbose_eval=False
)

y_pred_final = final_model.predict(dtest)
auc_final = roc_auc_score(y_test, y_pred_final)

print(f"\nTest AUC (final XGBoost): {auc_final:.3f}")


import pickle

with open('xgb_model.bin', 'wb') as f_out:
    pickle.dump((dv_full, final_model), f_out)

print('The model is saved to xgb_model.bin')

