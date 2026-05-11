from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel, Field

app = FastAPI(
    title="Buzzer Detection API",
    description="""
## Model Hybrid Machine Learning untuk Deteksi Buzzer di Media Sosial

API ini memprediksi apakah sebuah akun terindikasi **buzzer** atau **pengguna asli**
berdasarkan metadata dan pola interaksi pengguna di Reddit.

### Cara Penggunaan
1. Masukkan data metadata akun pada endpoint `/predict`
2. Model akan mengembalikan prediksi beserta confidence score

### Kategori Hasil
- **Pengguna Asli** → akun normal
- **Akun Buzzer** → akun terindikasi buzzer (confidence ≥ 0.30)
    """,
    version="1.0.0",
    contact={
        "name": "Albert Jonathan",
        "email": "Albertjonathan203@yahoo.com",
    },
)

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    score: float = Field(..., example=-5, description="Skor komentar pengguna")
    controversiality: float = Field(..., example=1, description="0 = tidak kontroversial, 1 = kontroversial")
    user_comment_karma: float = Field(..., example=10, description="Total karma komentar pengguna")
    user_link_karma: float = Field(..., example=0, description="Total karma link pengguna")
    user_total_karma: float = Field(..., example=10, description="Total karma keseluruhan pengguna")
    account_age_days: float = Field(..., example=2, description="Umur akun dalam hari")
    comment_length: float = Field(..., example=5, description="Panjang komentar dalam jumlah kata")

    class Config:
        json_schema_extra = {
            "examples": {
                "Pengguna Normal": {
                    "summary": "Contoh akun normal",
                    "value": {
                        "score": 10,
                        "controversiality": 0,
                        "user_comment_karma": 500,
                        "user_link_karma": 100,
                        "user_total_karma": 600,
                        "account_age_days": 365,
                        "comment_length": 17
                    }
                },
                "Akun Buzzer": {
                    "summary": "Contoh akun buzzer",
                    "value": {
                        "score": 1000,
                        "controversiality": 1,
                        "user_comment_karma": 600000,
                        "user_link_karma": 100000,
                        "user_total_karma": 1400000,
                        "account_age_days": 3000,
                        "comment_length": 30
                    }
                }
            }
        }

class PredictionResult(BaseModel):
    prediction: int = Field(..., description="0 = Pengguna Asli, 1 = Akun Buzzer")
    label: str = Field(..., description="Label hasil prediksi")
    confidence: float = Field(..., description="Tingkat keyakinan model (0.0 - 1.0)")

@app.get("/", tags=["Status"])
def root():
    return {"status": "Buzzer Detection API is running!", "version": "1.0.0"}

@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["Prediksi"],
    summary="Deteksi Buzzer",
    description="Memprediksi apakah akun terindikasi buzzer berdasarkan metadata pengguna."
)
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
