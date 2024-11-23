from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 模型加载
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 特征信息
features_info = [
    {"name": "Antiplatelet Drug Discontinuation", "unit": "", "options": ["Short discontinuation", "Delayed discontinuation"]},
    {"name": "NT proBNP", "unit": "pg/ml"},
    {"name": "APTT", "unit": "s"},
    {"name": "Hb", "unit": "g/L"},
    {"name": "Urea", "unit": "mmol/L"},
    {"name": "cTnT", "unit": "ng/mL"},
    {"name": "TBIL", "unit": "μmol/L"},
    {"name": "eGFR", "unit": "ml/min/1.73m²"},
    {"name": "Fibrinogen", "unit": "mg/dL"},
    {"name": "INR", "unit": ""},
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", top_10_features=features_info, zip=zip)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 从表单中获取数据
        features = [float(request.form[f'feature_{i}']) for i in range(10)]
        features = np.array(features).reshape(1, -1)

        # 执行预测
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]

        # 渲染结果页面
        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=float(probability),
            top_10_features=features_info,
            input_values=features[0].tolist()
        )
    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)