from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# --- Load model ---
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

threshold = model_data["threshold"]
print(f"Model loaded. Threshold: {threshold}")

app = FastAPI(title="Transaction ML API")

class Transaction(BaseModel):
    user_id_scaled:   float
    amount_scaled:    float
    merchant_encoded: int
    hour:             int
    day_of_week:      int

@app.get("/")
def root():
    return {"status": "running", "model": "SimpleThresholdModel"}

@app.post("/predict")
def predict(data: Transaction):
    prediction = 1 if data.day_of_week >= threshold else 0
    return {
        "prediction": prediction,
        "label": "weekend" if prediction == 1 else "weekday",
        "day_of_week_received": data.day_of_week
    }