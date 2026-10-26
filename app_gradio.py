from pathlib import Path
import joblib
import gradio as gr
import numpy as np

MODEL_PATH = Path("model.joblib")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[
        sepal_length, sepal_width, petal_length, petal_width
    ]])
    prediction_index = int(model.predict(features)[0])
    return target_names[prediction_index]

demo = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Number(label="Sepal length (cm)", value=5.1, minimum=0.01),
        gr.Number(label="Sepal width (cm)", value=3.5, minimum=0.01),
        gr.Number(label="Petal length (cm)", value=1.4, minimum=0.01),
        gr.Number(label="Petal width (cm)", value=0.2, minimum=0.01),
    ],
    outputs=gr.Textbox(label="Predicted species"),
    title="Iris Flower Predictor",
    description="A small demo showing how a saved model can be used in an app.",
)

if __name__ == "__main__":
    demo.launch()
