import streamlit as st
import joblib

st.title("ML Model Inference App")
model_version = st.selectbox("Select model version", ["LogisticRegression", "RandomForest"])

import logging

@st.cache_resource
def load_model(version):
    try:
        model_path = f"models/{version}.pkl"
        model = joblib.load(model_path)
        logging.info(f"Loaded model version {version}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {version}: {e}")
        st.error("Failed to load model.")
        return None
    
input_val = st.number_input("Enter input value:")
model = load_model(model_version)

if model and st.button("Predict"):
    try:
        pred = model.predict([[input_val]])
        st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        logging.exception("Prediction failed")
        st.error("Prediction error occurred.")

import logging

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

