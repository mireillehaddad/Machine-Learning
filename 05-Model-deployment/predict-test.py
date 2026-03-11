#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '0')


# In[2]:


import requests


# In[3]:


url = 'http://localhost:9696/predict'


# In[4]:


customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 1,
  "monthlycharges": 29.85,
  "totalcharges": 29.85
}


# In[5]:


customer


# In[6]:


response = requests.post(url, json=customer)

print(response.status_code)
print(response.text)


# In[7]:


requests.post(url,json=customer).json()


# In[9]:


response = requests.post(url, json=customer).json()


# In[10]:


if response['churn'] == True:
    print('sending promo email to %s' % ('xyz-123'))

