# need pip install requests
# Run when port is live either by docker or predict.py
import requests

url = 'http://localhost:9696/predict'
## url = 'https://churn-api-1082768418869.northamerica-northeast1.run.app/predict'##for GCP

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

response = requests.post(url, json=customer).json()

print(response)

if response['churn']:
    print('sending promo email to xyz-123')