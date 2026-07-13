from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "artifacts" / "wellbeing_risk_model.joblib"

RAW_FEATURES = [
    "sleep",
    "stress",
    "mood",
    "energy",
    "screenTime",
    "workHours",
    "activity",
    "social",
]
DOMAIN_COLUMNS = [
    "physical_score",
    "mental_score",
    "social_score",
    "occupational_score",
    "digital_score",
]
MODEL_BASE_COLUMNS = RAW_FEATURES + DOMAIN_COLUMNS + ["wellbeing_score"]


class DailyEntry(BaseModel):
    student_id: str = Field(..., examples=["u01"])
    date: str = Field(..., examples=["2013-04-04"])
    sleep: float = Field(..., ge=0, le=14)
    stress: float = Field(..., ge=1, le=10)
    mood: float = Field(..., ge=1, le=10)
    energy: float = Field(..., ge=1, le=10)
    screenTime: float = Field(..., ge=0, le=24)
    workHours: float = Field(..., ge=0, le=24)
    activity: float = Field(..., ge=0, le=300)
    social: float = Field(..., ge=1, le=10)


class PredictionRequest(BaseModel):
    entries: list[DailyEntry]
    student_id: str | None = Field(None, examples=["u01"])


class PredictionResponse(BaseModel):
    student_id: str
    latest_date: str
    prediction: int
    label: str
    probability: float
    threshold: float
    model_type: str
    task: str
    metrics: dict[str, Any]


app = FastAPI(
    title="Personal Wellbeing Risk API",
    description="FastAPI backend for next-day wellbeing-risk prediction.",
    version="1.0.0",
)


def clamp(series, lower=0, upper=100):
    return series.clip(lower=lower, upper=upper)


def add_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    sleep_score = clamp((scored["sleep"] / 8) * 100)
    stress_score = clamp(100 - (scored["stress"] - 1) * 11.1)
    mood_score = clamp(scored["mood"] * 10)
    energy_score = clamp(scored["energy"] * 10)
    social_score = clamp(scored["social"] * 10)
    activity_score = clamp((scored["activity"] / 45) * 100)
    screen_score = 100 - clamp((scored["screenTime"] - 5).clip(lower=0) * 10, 0, 60)
    work_score = 100 - clamp((scored["workHours"] - 8).clip(lower=0) * 12, 0, 60)

    scored["physical_score"] = sleep_score * 0.45 + activity_score * 0.30 + energy_score * 0.25
    scored["mental_score"] = stress_score * 0.50 + mood_score * 0.35 + energy_score * 0.15
    scored["social_score"] = social_score
    scored["occupational_score"] = work_score * 0.55 + stress_score * 0.30 + energy_score * 0.15
    scored["digital_score"] = screen_score * 0.70 + sleep_score * 0.15 + stress_score * 0.15
    scored["wellbeing_score"] = (
        scored["physical_score"] * 0.24
        + scored["mental_score"] * 0.28
        + scored["social_score"] * 0.14
        + scored["occupational_score"] * 0.17
        + scored["digital_score"] * 0.17
    )
    return scored


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy().sort_values(["student_id", "date"]).reset_index(drop=True)
    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek
    feature_df["days_since_student_start"] = (
        feature_df["date"] - feature_df.groupby("student_id")["date"].transform("min")
    ).dt.days

    for column in MODEL_BASE_COLUMNS:
        grouped = feature_df.groupby("student_id")[column]
        for lag in [1, 2, 3]:
            feature_df[f"{column}_lag_{lag}"] = grouped.shift(lag)
        for window in [3, 7]:
            feature_df[f"{column}_rolling_{window}"] = grouped.transform(
                lambda values: values.rolling(window, min_periods=2).mean()
            )

    return feature_df


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def request_to_features(payload: PredictionRequest) -> tuple[pd.DataFrame, str]:
    if not payload.entries:
        raise HTTPException(status_code=400, detail="At least one daily entry is required.")

    df = pd.DataFrame([entry.dict() for entry in payload.entries])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["student_id", "date"] + RAW_FEATURES)
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid rows were provided.")

    selected_student = payload.student_id or str(df.sort_values("date").iloc[-1]["student_id"])
    df = add_model_features(add_domain_scores(df))
    return df, selected_student


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    artifact = load_model_artifact()
    metadata = artifact.get("metadata", {})
    feature_columns = metadata.get("features", [])
    pipeline = artifact.get("pipeline")

    if pipeline is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Model artifact is missing pipeline metadata.")

    feature_df, selected_student = request_to_features(payload)
    student_rows = feature_df[feature_df["student_id"] == selected_student].sort_values("date")
    if student_rows.empty:
        raise HTTPException(status_code=400, detail=f"No rows found for student_id={selected_student}.")

    missing_features = [column for column in feature_columns if column not in student_rows.columns]
    if missing_features:
        raise HTTPException(status_code=500, detail={"missing_features": missing_features})

    latest = student_rows.iloc[[-1]]
    probability = float(pipeline.predict_proba(latest[feature_columns])[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "student_id": selected_student,
        "latest_date": latest.iloc[0]["date"].date().isoformat(),
        "prediction": prediction,
        "label": "Elevated risk" if prediction else "Lower risk",
        "probability": round(probability, 4),
        "threshold": 0.5,
        "model_type": metadata.get("model_type", "unknown"),
        "task": metadata.get("task", "next-day wellbeing risk prediction"),
        "metrics": metadata.get("metrics", {}),
    }
