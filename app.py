from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# Top 10 features and their units
features_info = [
    "NT proBNP", "APTT", "Hb", "Urea", "cTnT",
    "TBIL", "eGFR", "Fibrinogen", "INR"
]
units = [
    "(pg/ml)", "(s)", "(g/L)", "(mmol/L)", "(ng/mL)",
    "(μmol/L)", "(ml/min/1.73m²)", "(mg/dL)", ""
]

@app.route("/")
def index():
    return render_template("index.html", top_10_features=features_info, units=units)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[f'feature_{i}']) for i in range(len(features_info) + 1)]
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0, 1]
        return f"Prediction: {prediction}, Probability: {probability:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
