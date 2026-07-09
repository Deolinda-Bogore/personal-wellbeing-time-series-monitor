#!/usr/bin/env python3
"""
Train a small baseline wellbeing-risk model from an app-ready student-level CSV.

The script intentionally uses only Python's standard library so the project can
run on a fresh machine before heavier ML dependencies are added.

Usage:
  python3 models/train_baseline.py data/studentlife_wellbeing.csv
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path


FEATURES = [
    "sleep",
    "stress",
    "mood",
    "energy",
    "screenTime",
    "workHours",
    "activity",
    "social",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def domain_scores(row: dict[str, float]) -> dict[str, int]:
    sleep_score = clamp((row["sleep"] / 8) * 100, 0, 100)
    stress_score = 100 - (row["stress"] - 1) * 11.1
    mood_score = row["mood"] * 10
    energy_score = row["energy"] * 10
    social_score = row["social"] * 10
    activity_score = clamp((row["activity"] / 45) * 100, 0, 100)
    screen_score = 100 - clamp(max(row["screenTime"] - 5, 0) * 10, 0, 60)
    work_score = 100 - clamp(max(row["workHours"] - 8, 0) * 12, 0, 60)

    return {
        "physical": round(sleep_score * 0.45 + activity_score * 0.3 + energy_score * 0.25),
        "mental": round(stress_score * 0.5 + mood_score * 0.35 + energy_score * 0.15),
        "social": round(social_score),
        "occupational": round(work_score * 0.55 + stress_score * 0.3 + energy_score * 0.15),
        "digital": round(screen_score * 0.7 + sleep_score * 0.15 + stress_score * 0.15),
    }


def wellbeing_score(row: dict[str, float]) -> int:
    scores = domain_scores(row)
    return round(
        scores["physical"] * 0.24
        + scores["mental"] * 0.28
        + scores["social"] * 0.14
        + scores["occupational"] * 0.17
        + scores["digital"] * 0.17
    )


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            try:
                row = {feature: float(raw[feature]) for feature in FEATURES}
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No valid rows found in {path}")
    return rows


def normalize(rows: list[dict[str, float]]) -> tuple[list[list[float]], dict[str, dict[str, float]]]:
    stats = {}
    for feature in FEATURES:
        values = [row[feature] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance) or 1
        stats[feature] = {"mean": mean, "std": std}

    matrix = []
    for row in rows:
        matrix.append([(row[feature] - stats[feature]["mean"]) / stats[feature]["std"] for feature in FEATURES])
    return matrix, stats


def sigmoid(value: float) -> float:
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1 / (1 + math.exp(-value))


def train_logistic_regression(x_train: list[list[float]], y_train: list[int], epochs: int = 800, lr: float = 0.08):
    weights = [0.0 for _ in FEATURES]
    bias = 0.0

    for _ in range(epochs):
        grad_w = [0.0 for _ in FEATURES]
        grad_b = 0.0
        for x_row, label in zip(x_train, y_train):
            pred = sigmoid(sum(w * x for w, x in zip(weights, x_row)) + bias)
            error = pred - label
            for index, value in enumerate(x_row):
                grad_w[index] += error * value
            grad_b += error

        n = len(x_train)
        weights = [weight - lr * grad / n for weight, grad in zip(weights, grad_w)]
        bias -= lr * grad_b / n

    return weights, bias


def evaluate(weights: list[float], bias: float, x_rows: list[list[float]], labels: list[int]) -> dict[str, float]:
    if not x_rows:
        return {"accuracy": 0, "precision": 0, "recall": 0}

    predictions = [1 if sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) >= 0.5 else 0 for row in x_rows]
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)

    return {
        "accuracy": round((tp + tn) / len(labels), 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 models/train_baseline.py data/studentlife_wellbeing.csv")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    rows = load_rows(input_path)
    labels = [1 if wellbeing_score(row) < 70 else 0 for row in rows]
    matrix, stats = normalize(rows)

    combined = list(zip(matrix, labels))
    random.Random(42).shuffle(combined)
    split = max(1, int(len(combined) * 0.8))
    train = combined[:split]
    test = combined[split:] or combined[:]

    x_train = [row for row, _ in train]
    y_train = [label for _, label in train]
    x_test = [row for row, _ in test]
    y_test = [label for _, label in test]

    weights, bias = train_logistic_regression(x_train, y_train)
    metrics = evaluate(weights, bias, x_test, y_test)

    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "model_type": "standard-library-logistic-regression",
        "task": "predict wellbeing risk where risk = wellbeing_score < 70",
        "features": FEATURES,
        "normalization": stats,
        "weights": weights,
        "bias": bias,
        "metrics": metrics,
        "rows_used": len(rows),
    }

    model_path = output_dir / "baseline_risk_model.json"
    metrics_path = output_dir / "baseline_metrics.json"
    model_path.write_text(json.dumps(model, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Rows used: {len(rows)}")
    print(f"Risk examples: {sum(labels)}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
