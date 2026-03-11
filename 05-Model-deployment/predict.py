import pickle
from pprint import pprint

# Load the model
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print("Model loaded successfully")

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

def predict(customer):
    # Transform input
    X = dv.transform([customer])
    # Predict churn probability
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred


y_pred = predict(customer) 


print("Customer data:")
pprint(customer)
print(f"Churn probability: {y_pred:.3f}")

# Decision
result = {
    "churn_probability": round(float(y_pred), 3),
    "churn": bool(y_pred >= 0.5)
}

