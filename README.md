# Machine-Learning
Setting Up the enviroment
```bash
pip install jupyter numpy pandas scikit-learn seaborn
```
start Jupyter notebook
```bash
jupyter notebook
```
 jupyter nbconvert --to script 02-price-prediction-live.ipynb 


python -m venv .venv

source .venv/bin/activate
======================================================
save final mode:

import pickle

with open('xgb_model.bin', 'wb') as f_out:
    pickle.dump((dv_full, final_model), f_out)

print('The model is saved to xgb_model.bin')

======================================================
load it with:
import pickle

with open('xgb_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

======================================================

Virtual enviroment:

.venv\Scripts\activate


For virtual enviroment requirements file

```
pip freeze > requirements.txt
```
To install packages from requirements file: in case of repository cloning 

```
pip install -r requirements.txt
```


Recap after git cloning

```
git clone <repo_url>
cd repo

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

======================================================

# 6. Enviroment management: Docker

- Why we need Docker
- Running a Python image with docker
- Dockerfile
- Building a docker image
- Running a docker image


 Start a Python container Docker  and open an interactive Python session inside it.

```
docker run -it --rm python:3.8.12-slim

or  

docker run -it --rm --entrypoint=bash python:3.8.12-slim

```

| Part                 | Meaning                        |
| -------------------- | ------------------------------ |
| `docker run`         | start a container              |
| `-it`                | interactive terminal           |
| `--rm`               | delete container when it stops |
| `python:3.8.12-slim` | use the Python Docker image    |


Every Docker image has a default command (entrypoint).
--entrypoint=bash

Every Docker image has a default command (entrypoint).

For the Python image the default entrypoint is essentially:

python

So normally if you run:

docker run python:3.8.12-slim

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
