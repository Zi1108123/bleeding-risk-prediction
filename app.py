from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 特征信息
features_info = [
    {"name": "Antiplatelet drug discontinuation", "unit": None},
    {"name": "NT ProBNP", "unit": "pg/ml"},
    {"name": "APTT", "unit": "s"},
    {"name": "Hb", "unit": "g/L"},
    {"name": "Urea", "unit": "mmol/L"},
    {"name": "cTnT", "unit": "ng/mL"},
    {"name": "TBIL", "unit": "μmol/L"},
    {"name": "eGFR", "unit": "ml/min/1.73m²"},
    {"name": "Fibrinogen", "unit": "mg/dL"},
    {"name": "INR", "unit": None}
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # 收集用户输入数据
            discontinuation = request.form.get("discontinuation")
            feature_values = [
                1 if discontinuation == "Short discontinuation" else 2
            ]
            for i in range(1, len(features_info)):
                feature_values.append(float(request.form[f"feature_{i}"]))

            # 转为 NumPy 数组
            feature_array = np.array(feature_values).reshape(1, -1)

            # 预测
            probability = model.predict_proba(feature_array)[0, 1]
            prediction = model.predict(feature_array)[0]

            # 返回预测结果
            return render_template(
                "index.html",
                features_info=features_info,
                prediction=prediction,
                probability=probability,
                input_values=feature_values,
                enumerate=enumerate
            )
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("index.html", features_info=features_info, enumerate=enumerate)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
