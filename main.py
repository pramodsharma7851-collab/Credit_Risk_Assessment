from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ml_model = {}


def load_artifacts():
    if "model" not in ml_model:
        ml_model["model"] = joblib.load("credit_risk_model.pkl")
    if "threshold" not in ml_model:
        if os.path.exists("best_threshold.pkl"):
            try:
                ml_model["threshold"] = float(joblib.load("best_threshold.pkl"))
            except Exception:
                ml_model["threshold"] = 0.50
        else:
            ml_model["threshold"] = 0.50


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load model and threshold on startup
    load_artifacts()
    yield
    ml_model.clear()


app = FastAPI(title="Credit Risk Assessment API", lifespan=lifespan)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input validation schema
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post("/predict")
def predict(data: LoanApplication):
    # Ensure artifacts are loaded even if lifespan was not triggered
    load_artifacts()

    input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    input_df = pd.DataFrame([input_dict])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = float(ml_model.get("threshold", 0.50))
    prediction = int(probability >= threshold)

    return {
        "default_probability": round(probability, 4),
        "default_prediction": prediction,
        "threshold": round(threshold, 4),
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Mount the frontend web UI
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
