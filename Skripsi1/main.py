from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Buzzer Detection API",
    description="Hybrid ML untuk Deteksi Buzzer di Media Sosial"
)

model = joblib.load("model.pkl")

class UserFeatures(BaseModel):
    score: float
    controversiality: float
    user_comment_karma: float
    user_link_karma: float
    user_total_karma: float
    account_age_days: float
    comment_length: float

@app.get("/")
def root():
    return {
        "status": "active",
        "message": "Buzzer Detection API is running!",
        "model": "Hybrid Stacking Ensemble"
    }

@app.post("/predict")
def predict(data: UserFeatures):
    features = np.array([[
        data.score,
        data.controversiality,
        data.user_comment_karma,
        data.user_link_karma,
        data.user_total_karma,
        data.account_age_days,
        data.comment_length
    ]])

    proba = model.predict_proba(features)[0][1]
    label = "Terindikasi Buzzer" if proba >= 0.50 else "Normal"

    return {
        "prediction": label,
        "buzzer_probability": round(float(proba), 4),
        "normal_probability": round(1 - float(proba), 4),
        "threshold": 0.50
    }
