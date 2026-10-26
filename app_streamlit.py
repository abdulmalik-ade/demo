import joblib
import numpy as np
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("model.joblib")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

st.set_page_config(page_title="Flower Predictor", page_icon="🌸")
st.title("🌸 Iris Flower Predictor")
st.write(
    "Enter four flower measurements in centimetres. "
    "This is a learning demo, not a scientific identification tool."
)

sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict species"):
    features = np.array([[
        sepal_length, sepal_width, petal_length, petal_width
    ]])
    prediction_index = int(model.predict(features)[0])
    st.success(f"Predicted species: {target_names[prediction_index]}")
