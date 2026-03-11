# 2 Saving and loading the model

- Saving the model to pickle
- Loading the model from pickle
- Turning our notebook into a python script




```
jupyter nbconvert --to script train-churn-model.ipynb 
```

Flask

for ping.py

def ping():
    return "PONG"
    

```
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ ipython
In [1]: import ping

In [2]: ping.ping()
Out[2]: 'PONG'

```
to stop ipython: ctrl+Z or ctrl+C
Install Flask:

```
pip install flask

```
# 3. Web services: Introduction to Flask
- Writing a simple ping/pong app
- Querying it with 'Curl' and browser


Our new script: ping.py
```
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

```

```

python ping.py

```

```
curl http://127.0.0.1:9696/ping
```

# 4. Serving the churn model with Flask

- Wrapping the predict script into a Flask app
- Querying it with 'requests'
- Preparing for production: gunicorn
- Running it on Windows with waitress

Customer example:

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

Run 

```
python predict-deploy.py

curl http://127.0.0.1:9697/predict

05-predict-test.ipynb
```
pip install gunicorn

In terminal

gunicorn --bind 0.0.0.0:9696 predict-deploy:app
