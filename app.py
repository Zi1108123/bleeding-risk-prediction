from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 顶部10个特征信息
features_info = [
    "Antiplatelet drug discontinuation",
    "NT ProBNP",
    "APTT",
    "TBIL",
    "Hb",
    "cTnT",
    "eGFR",
    "Urea",
    "Fibrinogen",
    "INR"
]
units = [
    "",
    "pg/ml",
    "s",
    "μmol/L",
    "g/L",
    "ng/mL",
    "ml/min/1.73m²",
    "mmol/L",
    "mg/dL",
    ""
]

@app.route('/')
def index():
    return render_template("index.html", features_info=features_info, units=units)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # 获取输入数据
        input_values = request.form.to_dict(flat=False)
        input_values = [float(input_values[f"feature_{i}"][0]) for i in range(10)]

        # 预测
        prediction = model.predict(np.array(input_values).reshape(1, -1))[0]
        probability = model.predict_proba(np.array(input_values).reshape(1, -1))[0, 1]

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(probability * 100, 2),
            features=features_info,
            input_values=input_values
        )
    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
