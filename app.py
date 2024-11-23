from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 定义特征名称和单位
features_info = [
    ("Antiplatelet drug discontinuation", ""),  # 特殊处理为按钮
    ("NT ProBNP", "pg/ml"),
    ("APTT", "s"),
    ("Hb", "g/L"),
    ("Urea", "mmol/L"),
    ("cTnT", "ng/mL"),
    ("TBIL", "μmol/L"),
    ("eGFR", "ml/min/1.73m²"),
    ("Fibrinogen", "mg/dL"),
    ("INR", ""),  # 无单位
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # 从表单收集数据
            features = []
            # 第一个变量：Antiplatelet drug discontinuation
            discontinuation = request.form.get("discontinuation")
            features.append(1 if discontinuation == "Short discontinuation" else 2)

            # 收集其余数值型变量
            for i in range(1, len(features_info)):
                feature_name, _ = features_info[i]
                value = float(request.form[f"feature_{i}"])
                features.append(value)

            # 转换为数组
            features = np.array(features).reshape(1, -1)

            # 执行预测
            probability = model.predict_proba(features)[0, 1]
            prediction = model.predict(features)[0]

            return render_template(
                "result.html",
                prediction=prediction,
                probability=round(probability, 2),
            )
        except Exception as e:
            return render_template("error.html", error=str(e))

    return render_template("index.html", features_info=features_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
