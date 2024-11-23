from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 特征信息
features_info = [
    ("Antiplatelet Drug Discontinuation", "Short Discontinuation (≤3 days) / Delayed Discontinuation (>3 days)", ""),
    ("NT ProBNP", "N-terminal pro-B-type natriuretic peptide", "pg/ml"),
    ("APTT", "Activated Partial Thromboplastin Time", "s"),
    ("Hb", "Hemoglobin", "g/L"),
    ("Urea", "Urea", "mmol/L"),
    ("cTnT", "Cardiac Troponin T", "ng/mL"),
    ("TBIL", "Total Bilirubin", "μmol/L"),
    ("eGFR", "Estimated Glomerular Filtration Rate", "ml/min/1.73m²"),
    ("Fibrinogen", "Fibrinogen", "mg/dL"),
    ("INR", "International Normalized Ratio", "")
]

@app.route('/')
def index():
    return render_template("index.html", top_10_features=features_info)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # 获取用户输入
        features = [float(request.form[f'feature_{i}']) for i in range(len(features_info))]
        features = np.array(features).reshape(1, -1)

        # 预测
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]

        # 返回结果
        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability
        )
    except Exception as e:
        return render_template("error.html", error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
