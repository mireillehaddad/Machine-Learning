#!/usr/bin/env python
# coding: utf-8

# 2. Machine Learning for regression
# 

# In[ ]:





# In[377]:


import pandas as pd
import numpy as np


# 2.2 Data preparation

# In[378]:


#data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'


# In[379]:


#!wget $data 


# In[380]:


df = pd.read_csv('data.csv')
df.head(5)


# In[381]:


df.columns =df.columns.str.lower().str.replace(' ', '_')
df.head(5)


# In[382]:


strings= list(df.dtypes[df.dtypes == 'string'].index)
strings


# In[383]:


for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')


# In[384]:


df.head()


# In[385]:


df.dtypes


# Exploratory data analysis

# In[386]:


df.columns


# In[387]:


for col in df.columns:
    print(col)
    print(df[col].unique()[:5])  # Show first 5 unique values for each column
    print(df[col].nunique())  # Show first 5 unique values for each column 
    print()



# Distribution of price

# In[388]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[389]:


sns.histplot(df.msrp, bins=50)


# In[390]:


sns.histplot(df.msrp[df.msrp < 100000], bins=50)


# In[391]:


np.log1p([0, 1, 10, 1000, 100000])


# In[392]:


price_logs = np.log1p(df.msrp)
price_logs


# In[393]:


sns.histplot(price_logs, bins=50)


# In[394]:


df.isnull().sum()


# 2.4 Setting up the validation frame work

# In[395]:


n = len(df)
n_val =int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test


# In[ ]:





# In[ ]:





# In[396]:


df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]


# In[397]:


df_val


# In[398]:


idx = np.arange(n)


# In[399]:


np.random.seed(2)
np.random.shuffle(idx)
idx


# In[400]:


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]   


# In[401]:


df_train.head()


# In[402]:


len(df_train), len(df_val), len(df_test)


# In[403]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[404]:


y_train =np.log1p(df_train.msrp.values)
y_val =np.log1p(df_val.msrp.values)
y_test =np.log1p(df_test.msrp.values)


# In[405]:


del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


# In[406]:


len(y_train), len(y_val), len(y_test)


# ## Linear Regression

# Training  Linear regression 

# In[407]:


df_train.iloc[10]


# In[408]:


xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]


# In[409]:


def linear_regression(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]

    return pred


# In[410]:


xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]


# In[411]:


np.expm1(12.312)


# In[412]:


def dot(xi, w):
    n = len(xi)

    res = 0.0

    for j in range(n):
        res = res + xi[j] * w[j]

    return res


# In[413]:


def linear_regression(xi):
    return w0 + dot(xi, w)


# In[414]:


w_new = [w0] + w


# In[415]:


w_new


# In[416]:


def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)


# In[417]:


linear_regression(xi)


# In[418]:


w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w


# In[419]:


x1  = [1, 148, 24, 1385]
x2  = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X)
X


# In[420]:


def linear_regression(X):
    return X.dot(w_new)


# In[421]:


linear_regression(X)


# In[422]:


def train_linear_regression(X, y):
    pass


# In[423]:


X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = np.array(X)
X


# In[424]:


ones = np.ones(X.shape[0])
ones


# In[425]:


X = np.column_stack([ones, X])


# In[426]:


y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]


# In[427]:


XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
w_full = XTX_inv.dot(X.T).dot(y)


# In[428]:


w0 = w_full[0]
w = w_full[1:]


# In[429]:


w0, w


# In[430]:


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# Car price baseline model

# In[431]:


df_train.columns


# In[432]:


base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)


# In[433]:


w0


# In[434]:


w


# In[435]:


sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_train, color='blue', alpha=0.5, bins=50)


# RMSE

# In[436]:


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# In[437]:


rmse(y_train, y_pred)


# Validating the model

# In[438]:


base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)


# In[439]:


def prepare_X(df):
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
            'city_mpg', 'popularity']

    X = df[base].fillna(0).values

    return X


# In[440]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0+X_val.dot(w)
rmse(y_val, y_pred)


# Simple feature engineering

# In[441]:


def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
            'city_mpg', 'popularity']
    features = base + ['age']
    df_num = df[features]
    df_num = df_num.fillna(0)


    X = df_num .values

    return X


# In[442]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0+X_val.dot(w)
rmse(y_val, y_pred)


# In[443]:


sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)


# In[ ]:





# 2.12 Categorical variables

# In[444]:


categorical_columns = [
    'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
    'vehicle_size', 'vehicle_style']

categorical = {}

for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)


# In[445]:


def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df['year']
    features.append('age')

    for v in [2, 3, 4]:
        df['num_doors_%d' % v] = (df.number_of_doors == v).astype(int)
        features.append('num_doors_%d' % v)

#    for name, values in categorical.items():
#        for value in values:
#            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
#            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


# In[446]:


prepare_X(df_train)


# In[ ]:





# In[447]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[448]:


w0, w


# In[449]:


sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)


# In[450]:


makes =list(df.make.value_counts().head().index)


# In[451]:


def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df['year']
    features.append('age')

    for v in [2, 3, 4]:
        df['num_doors_%d' % v] = (df.number_of_doors == v).astype(int)
        features.append('num_doors_%d' % v)

    for v in makes:
        df['make_%s' % v] = (df.make == v).astype(int)
        features.append('make_%s' % v)



#    for name, values in categorical.items():
#        for value in values:
#            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
#            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


# In[452]:


prepare_X(df_train)


# In[453]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[454]:


sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)


# In[455]:


categorical_columns = [
    'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
    'vehicle_size', 'vehicle_style']

categorical = {}

for c in categorical_columns:
    categorical[c] = list(df_train[c].value_counts().head().index)


# In[456]:


def prepare_X(df):
    df = df.copy()

    df['age'] = 2017 - df['year']
    features = base + ['age']

    for v in [2, 3, 4]:
        df['num_doors_%d' % v] = (df.number_of_doors == v).astype(int)
        features.append('num_doors_%d' % v)

    for name, values in categorical.items():
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


# In[457]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[458]:


w0, w


# In[ ]:





# 2.13 Regularization

# In[471]:


X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.00001],
]

X = np.array(X)
X


# In[472]:


y= [1, 2, 3, 1, 2, 3]


# In[473]:


XTX = X.T.dot(X)
XTX


# In[475]:


XTX_inv = np.linalg.inv(XTX)
XTX_inv


# In[476]:


XTX_inv.dot(X.T).dot(y)


# In[477]:


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])  # Add regularization
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


# In[ ]:


X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.001)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# In[480]:


for r in [0.0,0.00001, 0.0001, 0.001, 0.01, 1.0, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(f"r={r}, w0={w0}, score={score}")


# In[482]:


df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)


# In[485]:


X_full_train = prepare_X(df_full_train)


# In[486]:


X_full_train 


# In[487]:


y_full_train = np.concatenate([y_train, y_val])



# In[488]:


w0,w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)


# In[489]:


w


# In[490]:


X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score 


# In[491]:


df_test.iloc[20]


# In[494]:


car = df_test.iloc[20].to_dict()
car


# In[ ]:





# In[496]:


df_small = pd.DataFrame([car])
df_small


# In[498]:


X_small =prepare_X(df_small)


# In[500]:


y_pred = w0 + X_small.dot(w)
y_pred


# In[ ]:


np.expm1(y_pred)

