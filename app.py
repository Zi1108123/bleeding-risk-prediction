from flask import Flask, render_template, request
import joblib
import numpy as np

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型
model_path = 'bleeding_risk_model.pkl'
model = joblib.load(model_path)

# 变量信息
features_info = [
    ("Antiplatelet Drug Discontinuation", "Short discontinuation*", "Delayed discontinuation**"),
    ("NT proBNP (pg/ml)", "Numerical input", ""),
    ("APTT (s)", "Numerical input", ""),
    ("Hb (g/L)", "Numerical input", ""),
    ("Urea (mmol/L)", "Numerical input", ""),
    ("cTnT (ng/mL)", "Numerical input", ""),
    ("TBIL (μmol/L)", "Numerical input", ""),
    ("eGFR (ml/min/1.73m²)", "Numerical input", ""),
    ("Fibrinogen (mg/dL)", "Numerical input", ""),
    ("INR", "Numerical input", "")
]

@app.route('/')
def index():
    # 将 zip 传递给模板
    return render_template("index.html", top_10_features=features_info, zip=zip)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从表单中获取数据
        features = []
        for feature_info in features_info:
            feature_value = request.form.get(feature_info[0])
            if feature_info[0] == "Antiplatelet Drug Discontinuation":
                # 特殊处理选择框的值
                if feature_value == "Short discontinuation":
                    features.append(1)
                elif feature_value == "Delayed discontinuation":
                    features.append(2)
            else:
                features.append(float(feature_value))
        features = np.array(features).reshape(1, -1)

        # 执行预测
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]

        # 渲染结果页面
        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)