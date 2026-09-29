
import streamlit as st
import joblib
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "artifacts" / "iris_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="Iris ML Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Prediction")
st.write("Enter the flower measurements to predict its species.")

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

st.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict Flower Species"):
    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(features)[0]

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Species: {species[prediction]}")