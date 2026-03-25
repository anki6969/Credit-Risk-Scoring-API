from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Credit Risk Scoring API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Input schema
class LoanInput(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is running"}

@app.get("/health")
def health():
    return {"message": "System is Healthy"}

@app.post("/predict")
def predict(data: LoanInput):
    try:
        logging.info("Prediction request received")

        features = np.array([[
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History
        ]])

        features = scaler.transform(features)
        prob = model.predict_proba(features)[0][1]

        risk = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"

        return {
            "default_probability": round(float(prob), 3),
            "risk_level": risk
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}
