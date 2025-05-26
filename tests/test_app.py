import joblib
import pytest

def test_model_loading():
    model = joblib.load("models/LogisticRegression.pkl")
    assert hasattr(model, "predict")

def test_model_prediction():
    model = joblib.load("models/LogisticRegression.pkl")
    result = model.predict([[5]])
    assert isinstance(result[0], (int, float))
