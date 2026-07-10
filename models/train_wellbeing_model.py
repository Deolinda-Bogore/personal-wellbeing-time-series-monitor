#!/usr/bin/env python3
"""
Train a stronger next-day wellbeing-risk model.

This model predicts whether a student will be at risk on the next recorded day
using current-day signals, domain scores, lag features, and rolling averages.

Usage:
  python3 models/train_wellbeing_model.py data/studentlife_wellbeing.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


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

DOMAIN_FEATURES = [
    "physical_score",
    "mental_score",
    "social_score",
    "occupational_score",
    "digital_score",
    "wellbeing_score",
]


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


def add_time_series_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.copy().sort_values(["student_id", "date"]).reset_index(drop=True)
    feature_df["day_of_week"] = feature_df["date"].dt.dayofweek
    feature_df["days_since_student_start"] = (
        feature_df["date"] - feature_df.groupby("student_id")["date"].transform("min")
    ).dt.days

    generated_features = ["day_of_week", "days_since_student_start"]
    rolling_sources = RAW_FEATURES + DOMAIN_FEATURES

    for column in rolling_sources:
        grouped = feature_df.groupby("student_id")[column]
        for lag in [1, 2, 3]:
            name = f"{column}_lag_{lag}"
            feature_df[name] = grouped.shift(lag)
            generated_features.append(name)
        for window in [3, 7]:
            name = f"{column}_rolling_{window}"
            feature_df[name] = grouped.transform(lambda values: values.rolling(window, min_periods=2).mean())
            generated_features.append(name)

    feature_df["next_day_wellbeing_score"] = feature_df.groupby("student_id")["wellbeing_score"].shift(-1)
    feature_df["next_day_risk"] = (feature_df["next_day_wellbeing_score"] < 70).astype(int)
    feature_df = feature_df.dropna(subset=["next_day_wellbeing_score"]).reset_index(drop=True)

    feature_columns = RAW_FEATURES + DOMAIN_FEATURES + generated_features
    return feature_df, feature_columns


def metric_summary(y_true, y_pred, y_proba) -> dict[str, object]:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 3),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 models/train_wellbeing_model.py data/studentlife_wellbeing.csv")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["student_id", "date"] + RAW_FEATURES)
    df = add_domain_scores(df)
    model_df, feature_columns = add_time_series_features(df)

    groups = model_df["student_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(model_df[feature_columns], model_df["next_day_risk"], groups=groups))

    x_train = model_df.iloc[train_idx][feature_columns]
    y_train = model_df.iloc[train_idx]["next_day_risk"]
    x_test = model_df.iloc[test_idx][feature_columns]
    y_test = model_df.iloc[test_idx]["next_day_risk"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = metric_summary(y_test, y_pred, y_proba)

    classifier = model.named_steps["classifier"]
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "wellbeing_risk_model.joblib"
    metrics_path = output_dir / "wellbeing_risk_metrics.json"
    importance_path = output_dir / "wellbeing_feature_importance.csv"

    metadata = {
        "model_type": "random_forest_classifier",
        "task": "predict next-day wellbeing risk where next_day_wellbeing_score < 70",
        "input_csv": str(input_path),
        "rows_used": int(len(model_df)),
        "students_used": int(model_df["student_id"].nunique()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_students": int(model_df.iloc[train_idx]["student_id"].nunique()),
        "test_students": int(model_df.iloc[test_idx]["student_id"].nunique()),
        "positive_rate": round(float(model_df["next_day_risk"].mean()), 3),
        "features": feature_columns,
        "metrics": metrics,
    }

    joblib.dump({"pipeline": model, "metadata": metadata}, model_path)
    metrics_path.write_text(json.dumps(metadata, indent=2))
    feature_importance.to_csv(importance_path, index=False)

    print(f"Rows used: {metadata['rows_used']}")
    print(f"Students used: {metadata['students_used']}")
    print(f"Train/test rows: {metadata['train_rows']} / {metadata['test_rows']}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved feature importance: {importance_path}")
    print(json.dumps(metrics, indent=2))
    print("\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
