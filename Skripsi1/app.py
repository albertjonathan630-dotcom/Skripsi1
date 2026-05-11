from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Buzzer Detection API - Skripsi")

# Load model saat server start
model = joblib.load("model_buzzer.pkl")

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
    return {"message": "Buzzer Detection API is running!"}

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
        
