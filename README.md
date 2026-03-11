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


