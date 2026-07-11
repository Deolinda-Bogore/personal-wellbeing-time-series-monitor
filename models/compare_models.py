#!/usr/bin/env python3
"""
Compare next-day wellbeing-risk models on the same feature set.

This script trains a scikit-learn Logistic Regression and Random Forest on the
same next-day risk task, using a student-level split so test students are held
out from training.

Usage:
  python3 models/compare_models.py data/studentlife_wellbeing.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_wellbeing_model import RAW_FEATURES, add_domain_scores, add_time_series_features


def metrics_for(model, x_test, y_test) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
    }


def load_model_frame(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["student_id", "date"] + RAW_FEATURES)
    df = add_domain_scores(df)
    return add_time_series_features(df)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 models/compare_models.py data/studentlife_wellbeing.csv")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    model_df, feature_columns = load_model_frame(input_path)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        splitter.split(model_df[feature_columns], model_df["next_day_risk"], groups=model_df["student_id"])
    )

    x_train = model_df.iloc[train_idx][feature_columns]
    y_train = model_df.iloc[train_idx]["next_day_risk"]
    x_test = model_df.iloc[test_idx][feature_columns]
    y_test = model_df.iloc[test_idx]["next_day_risk"]

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
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
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_metrics = metrics_for(model, x_test, y_test)
        rows.append({"model": name, **model_metrics})

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": "predict next-day wellbeing risk where next_day_wellbeing_score < 70",
        "split": "GroupShuffleSplit by student_id",
        "rows_used": int(len(model_df)),
        "students_used": int(model_df["student_id"].nunique()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_students": int(model_df.iloc[train_idx]["student_id"].nunique()),
        "test_students": int(model_df.iloc[test_idx]["student_id"].nunique()),
        "features": feature_columns,
        "results": comparison.to_dict(orient="records"),
    }

    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"
    comparison.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison JSON: {json_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
