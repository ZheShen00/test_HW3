from streamlit.testing.v1 import AppTest
import pandas as pd

def test_app_loads_correctly():
    at = AppTest.from_file("app.py")
    at.run()
    assert at.title[0].value == "Cloud Type Classifier"
    assert len(at.selectbox) >= 1

def test_single_prediction_success():
    at = AppTest.from_file("app.py")
    at.run()

    # 设置模型（必须放前面，否则不会加载后续控件）
    at.selectbox[0].set_value("LogisticRegression")
    at.run()

    assert len(at.number_input) >= 3, "输入框未加载，模型可能加载失败"

    at.number_input[0].set_value(0.12)  # log_entropy
    at.number_input[1].set_value(0.34)  # IR_norm_range
    at.number_input[2].set_value(0.56)  # entropy_x_contrast

    # 触发预测按钮（按顺序点第一个）
    at.button[0].click()
    at.run()

    assert any("Prediction" in s.value for s in at.success)

import pandas as pd
import joblib

def test_batch_prediction_csv_simplified():
    # 用你准备的真实 CSV 文件（必须放在项目根目录）
    df = pd.read_csv("tests/test.csv", encoding="utf-8-sig")

    # 确保所需列存在
    required_cols = ["log_entropy", "IR_norm_range", "entropy_x_contrast"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # 加载模型并做预测
    model = joblib.load("models/LogisticRegression.pkl")
    preds = model.predict(df[required_cols])

    # 断言预测结果合理
    assert len(preds) == len(df)
    for p in preds:
        assert isinstance(p, (int, float, bool, str)), f"Invalid prediction type: {type(p)}"


