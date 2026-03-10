
```
5.2 Saving and loading the model

- Saving the model to pickle
- Loading the model from pickle
- Turning our notebook into a python script
```


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
to stop ipython: ctrl+Z
Install Flask:

```
pip install flask

```

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

