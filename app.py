from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the model
app = Flask(__name__)
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# Feature metadata
features_info = [
    {"name": "Antiplatelet Drug Discontinuation", "unit": ""},
    {"name": "NT proBNP", "unit": "pg/ml"},
    {"name": "APTT", "unit": "s"},
    {"name": "Hb", "unit": "g/L"},
    {"name": "Urea", "unit": "mmol/L"},
    {"name": "cTnT", "unit": "ng/mL"},
    {"name": "TBIL", "unit": "μmol/L"},
    {"name": "eGFR", "unit": "ml/min/1.73m²"},
    {"name": "Fibrinogen", "unit": "mg/dL"},
    {"name": "INR", "unit": ""}
]

@app.route("/")
def index():
    return render_template("index.html", top_10_features=features_info)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        features = [float(request.form[f"feature_{i}"]) for i in range(10)]
        features = np.array(features).reshape(1, -1)

        # Predict
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]

        # Render result
        return render_template(
            "result.html",
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        return render_template("error.html", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)