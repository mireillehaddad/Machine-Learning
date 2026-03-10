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