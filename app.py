
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_PATH = Path("churn_model_artifact.joblib")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts subscription customer churn probability and binary risk classification.",
    version="1.0.0",
)

artifact: dict = {}


@app.on_event("startup")
def load_artifact() -> None:
    global artifact
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at '{ARTIFACT_PATH}'. Run train_pipeline.py first."
        )
    artifact = joblib.load(ARTIFACT_PATH)


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Months the customer has stayed with the service")
    MonthlyCharges: float = Field(..., ge=0.0, description="Current monthly subscription charge")
    TotalCharges: Optional[float] = Field(
        default=None, ge=0.0, description="Cumulative charges billed to date"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month", description="Billing contract term"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic", description="Internet service type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether online security add-on is active"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether tech support add-on is active"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        default="Yes", description="Whether customer uses paperless billing"
    )
    PaymentMethod: str = Field(
        default="Electronic check", description="Customer payment method"
    )


class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., description="1 = likely to churn, 0 = likely to retain")
    risk_tier: Literal["Low", "Medium", "High"]
    threshold_used: float
    model_name: str


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_loaded": bool(artifact),
        "model_name": artifact.get("model_name"),
        "metrics": artifact.get("metrics"),
    }


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerFeatures) -> ChurnPredictionResponse:
    if not artifact or "model" not in artifact:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    model_name = str(artifact.get("model_name", "unknown"))

    input_df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(probability >= threshold)

    if probability >= 0.70:
        risk_tier: Literal["Low", "Medium", "High"] = "High"
    elif probability >= threshold:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    return ChurnPredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_tier=risk_tier,
        threshold_used=round(threshold, 4),
        model_name=model_name,
    )