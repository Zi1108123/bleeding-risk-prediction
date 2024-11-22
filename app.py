from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import pandas as pd

# 加载模型
model_path = os.path.join(os.path.dirname(__file__), "bleeding_risk_model.pkl")
model = joblib.load(model_path)

# 定义特征列表
selected_features = [
    "Antiplatelet drug discontinuation", "NT ProBNP", "APTT", "TBIL", "Hb",
    "cTnt", "eGRF", "Urea", "Fibrinogen", "INR"
]

# 初始化 Flask 应用
app = Flask(__name__)

# 向 Jinja2 模板添加 zip 函数
app.jinja_env.globals.update(zip=zip)


@app.route('/')
def home():
    """主页：动态显示模型重要性前10的变量"""
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)
    top_10_features = importance_df.head(10).to_dict(orient="records")
    return render_template('index.html', top_10_features=top_10_features)

@app.route('/web-predict', methods=['POST'])
def web_predict():
    """处理预测请求"""
    try:
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)
        top_10_features = importance_df.head(10)["Feature"].tolist()
        features = [float(request.form[f'feature_{i}']) for i in range(10)]
        features = np.array(features).reshape(1, -1)
        probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]
        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=float(probability),
            top_10_features=top_10_features,
            input_values=features[0].tolist()
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

