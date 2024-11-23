from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 定义重要性前 10 的特征
top_10_features = [
    "Antiplatelet Drug Discontinuation",
    "NT ProBNP",
    "APTT",
    "Hb",
    "Urea",
    "cTnT",
    "TBIL",
    "eGFR",
    "Fibrinogen",
    "INR"
]

# 定义变量的单位
units = {
    "NT ProBNP": "pg/ml",
    "APTT": "s",
    "Hb": "g/L",
    "Urea": "mmol/L",
    "cTnT": "ng/mL",
    "TBIL": "μmol/L",
    "eGFR": "ml/min/1.73m²",
    "Fibrinogen": "mg/dL",
    "INR": None  # 无单位
}

@app.route('/')
def index():
    return render_template('index.html', top_10_features=top_10_features, units=units)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取输入值
        features = [float(request.form[f'feature_{i}']) for i in range(10)]
        features = np.array(features).reshape(1, -1)

        # 模型预测
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]

        # 返回结果
        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=round(probability, 2),
            input_values=features[0].tolist()
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

