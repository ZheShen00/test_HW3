import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging

st.set_page_config(page_title="Cloud Type Predictor", layout="centered")
st.title("Cloud Type Classifier")

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# --- Config ---
FEATURES = ["log_entropy", "IR_norm_range", "entropy_x_contrast"]
MODEL_OPTIONS = ["LogisticRegression", "RandomForestClassifier"]

@st.cache_resource
def load_model(version):
    try:
        model_path = f"artifacts/{version}/model.pkl"
        model = joblib.load(model_path)
        logging.info(f"Loaded model: {version}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {version}: {e}")
        st.error("Model could not be loaded.")
        return None

# --- UI: Model Selection ---
model_version = st.selectbox("Select model version", MODEL_OPTIONS)
model = load_model(model_version)

# --- UI: Single Prediction ---
st.subheader("Single Prediction Input")

single_input = {}
for feat in FEATURES:
    single_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.5f")

if model and st.button("Predict Single"):
    try:
        input_df = pd.DataFrame([single_input])
        pred = model.predict(input_df)
        st.success(f"Prediction: {int(pred[0])}")
    except Exception as e:
        logging.exception("Single prediction failed")
        st.error("Error during prediction.")

# --- UI: Batch Prediction via File ---
st.subheader("Batch Prediction from CSV File")

uploaded_file = st.file_uploader("Upload CSV with required columns", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        missing = [col for col in FEATURES if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            preds = model.predict(df[FEATURES])
            df["prediction"] = preds
            st.success("Batch prediction complete.")
            st.dataframe(df)
            st.download_button("Download predictions as CSV", data=df.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        logging.exception("Batch prediction failed")
        st.error("Failed to process uploaded file.")

import logging

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

