from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 特征和单位定义
features_info = [
    {"name": "NT proBNP", "unit": "(pg/ml)"},
    {"name": "APTT", "unit": "(s)"},
    {"name": "Hb", "unit": "(g/L)"},
    {"name": "Urea", "unit": "(mmol/L)"},
    {"name": "cTnT", "unit": "(ng/mL)"},
    {"name": "TBIL", "unit": "(μmol/L)"},
    {"name": "eGFR", "unit": "(ml/min/1.73m²)"},
    {"name": "Fibrinogen", "unit": "(mg/dL)"},
    {"name": "INR", "unit": ""},
]

@app.route("/")
def index():
    # 传递特征和单位到模板
    return render_template("index.html", features_info=features_info)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取用户输入的特征值
        features = [float(request.form[f'feature_{i}']) for i in range(len(features_info))]
        # 模型预测
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0, 1]
        # 返回结果
        return f"Prediction: {prediction}, Probability: {probability:.2f}"
    except Exception as e:
        # 返回错误信息
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
