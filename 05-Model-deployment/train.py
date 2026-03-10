#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Load data
df = pd.read_csv('data-week-3.csv')

#parameters
C = 1.0
n_splits = 5

output_file = f'model_C={C}.bin'
output_file


# Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False)

# Include both object and string columns
categorical_columns = list(df.select_dtypes(include=['object', 'string']).columns)

# Clean text columns
for c in categorical_columns:
    df[c] = df[c].astype(str).str.lower().str.strip().str.replace(' ', '_', regex=False)

# Convert totalcharges
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

# Convert target
df['churn'] = (df['churn'] == 'yes').astype(int)

# Check target conversion
print(df['churn'].value_counts())
# expected:
# 0    5174
# 1    1869



# Split data
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=1,
    stratify=df['churn']
)


#Features
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


#Training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)

    return dv, model


#Prediction
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred



#Validation
print(f'Doing validation with C={C}')
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0
for train_idx, val_idx in kfold.split(df_full_train, df_full_train['churn'].values):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['churn'].values
    y_val = df_val['churn'].values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold = fold +1
print('Validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print(scores)



# Training the final model
print('Training the final model') 
dv, model = train(df_full_train, df_full_train['churn'].values, C=C)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(df_test['churn'].values, y_pred)
print(f'auc={auc}')



dv, model = train(df_full_train, df_full_train['churn'].values, C=C)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print('Test AUC:', auc)



scores


# 5.2 Saving and loading the model
# 
# - Saving the model to pickle
# - Loading the model from pickle
# - Turning our notebook into a python script
# 

# Save the model

#f_out = open(output_file, 'wb')
#pickle.dump((dv,model), f_out)
#f_out.close()



with open(output_file,'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f'The model is saved to{output_file}')




# Load the model




model_file = 'model_C=1.0.bin'



with open(model_file,'rb') as f_in:
    dv, model = pickle.load(f_in)



dv, model



customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}



X = dv.transform([customer])



model.predict_proba(X)[0,1]





