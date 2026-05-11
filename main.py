from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(
    title="Buzzer Detection API",
    description="Model Hybrid ML untuk Deteksi Buzzer di Media Sosial",
    version="1.0.0"
)

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    score: float
    controversiality: float
    user_comment_karma: float
    user_link_karma: float
    user_total_karma: float
    account_age_days: float
    comment_length: float

@app.get("/", tags=["Status"])
def root():
    return {"status": "Buzzer Detection API is running!", "version": "1.0.0"}

@app.post("/predict", tags=["Prediksi"], summary="Deteksi Buzzer")
def predict(data: InputData):
    features = np.array([[
        data.score, data.controversiality,
        data.user_comment_karma, data.user_link_karma,
        data.user_total_karma, data.account_age_days,
        data.comment_length
    ]])
    features_scaled = scaler.transform(features)
    proba      = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if proba >= 0.30 else 0
    label      = "Akun Buzzer" if prediction == 1 else "Pengguna Asli"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(float(proba), 4)
    }
