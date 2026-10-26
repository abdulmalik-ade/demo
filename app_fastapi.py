
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.joblib")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

app = FastAPI(title="Iris Prediction API", version="1.0.0")

class FlowerInput(BaseModel):
    # Field validates that values are numbers greater than zero
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

@app.get("/")
def home():
    return {"message": "Iris Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(flower: FlowerInput):
    features = np.array([[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]])

    prediction_index = int(model.predict(features)[0])
    return {
        "predicted_class_index": prediction_index,
        "predicted_species": target_names[prediction_index],
    }
