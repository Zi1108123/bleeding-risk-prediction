from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 手动将 zip 和 enumerate 注册为 Jinja2 的全局变量
app.jinja_env.globals.update(zip=zip, enumerate=enumerate)

# 加载模型
model_path = "bleeding_risk_model.pkl"
model = joblib.load(model_path)

# 特征信息和单位 - 必须与训练模型时的顺序完全一致
features_info = [
    "Antiplatelet drug discontinuation *",
    "NT ProBNP",
    "APTT",
    "TBIL",
    "Hb",
    "cTnT",
    "eGFR",
    "Urea",
    "Fibrinogen",
    "INR",
]
units = ["(Short or Delayed)", "pg/ml", "s", "μmol/L", "g/L", "ng/mL", "ml/min/1.73m²", "mmol/L", "mg/dL", ""]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            inputs = []
            for i in range(len(features_info)):
                if i == 0:
                    value = int(request.form.get(f"feature_{i}"))
                else:
                    value = float(request.form.get(f"feature_{i}"))
                inputs.append(value)

            inputs = np.array(inputs).reshape(1, -1)

            # 执行模型预测
            prediction = model.predict(inputs)[0]
            probability = model.predict_proba(inputs)[0][1]

            return render_template(
                "index.html",
                features_info=features_info,
                units=units,
                prediction=prediction,
                probability=round(probability, 3),
            )
        except Exception as e:
            # 如果出现错误，也需要传递 prediction 和 probability 为 None
            return render_template(
                "index.html",
                features_info=features_info,
                units=units,
                prediction=None,
                probability=None,
                error=str(e),
            )
    # 初始 GET 请求时传递空值
    return render_template(
        "index.html",
        features_info=features_info,
        units=units,
        prediction=None,
        probability=None,
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
