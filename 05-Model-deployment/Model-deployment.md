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



# 6. Enviroment management: Docker

- Why we need Docker
- Running a Python image with docker
- Dockerfile
- Building a docker image
- Running a docker image


 Start a Python container Docker  and open an interactive Python session inside it.

```
docker run -it --rm python:3.11-slim

or  

docker run -it --rm --entrypoint=bash python:3.11-slim

```

| Part                      | Meaning                        |
| ------------------------  | ------------------------------ |
| `docker run`              | start a container              |
| `-it`                     | interactive terminal           |
| `--rm`                    | delete container when it stops |
| `python:python:3.11-slim` | use the Python Docker image    |


Every Docker image has a default command (entrypoint).
--entrypoint=bash

Every Docker image has a default command (entrypoint).

For the Python image the default entrypoint is essentially:

python

So normally if you run:

docker run python:python:3.11-slim

you would enter:

Python interactive interpreter
>>>

But with:

--entrypoint=bash

you override the default command and instead start Bash.

So you get a Linux shell inside the container.

You will see something like:

root@13df9fff0603:/#

======================================================

In 
root@13df9fff0603:/# mkdir app
root@13df9fff0603:/# cd app/

we exit CTRL+Z

======================================================

docker build -t churn-test .

What happens internally

When you run the command Docker will:

1️⃣ Read the Dockerfile

2️⃣ Execute each instruction step by step

Example:

FROM python:3.8.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "predict-deploy.py"]

3️⃣ Build a Docker image


Project folder
   │
   ├── Dockerfile
   ├── churn.py
   ├── requirements.txt
   │
   ▼
docker build -t churn-test .
   │
   ▼
Docker Image (churn-test)
   │
   ▼
docker run churn-test
   │
   ▼
Running Container

======================================================

docker run -it --rm --entrypoint=bash churn-test


======================================================


docker run -it --rm -p 9696:9696 churn-test

pip install requests
======================================================

