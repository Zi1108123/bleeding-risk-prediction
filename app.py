from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
model = joblib.load("bleeding_risk_model.pkl")

# 定义变量名和单位
features_info = [
    {"name": "Antiplatelet drug discontinuation", "unit": "", "options": ["Short discontinuation (*)", "Delayed discontinuation (**)"]},
    {"name": "NT ProBNP", "unit": "pg/ml"},
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
    return render_template("index.html", features_info=features_info)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取用户输入
        inputs = []
        for i, feature in enumerate(features_info):
            if feature["name"] == "Antiplatelet drug discontinuation":
                inputs.append(int(request.form[f"feature_{i}"]))
            else:
                inputs.append(float(request.form[f"feature_{i}"]))

        # 转为 numpy 数组
        inputs = np.array(inputs).reshape(1, -1)

        # 预测
        probability = model.predict_proba(inputs)[0, 1]
        prediction = model.predict(inputs)[0]

        return render_template("result.html", prediction=int(prediction), probability=float(probability))
    except Exception as e:
        return render_template("error.html", error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
