#!/usr/bin/env python3
"""
Generate an EDA report for the app-ready wellbeing CSV.

Usage:
  python3 scripts/eda_processed_data.py data/studentlife_wellbeing.csv

Outputs:
  reports/eda_report.md
  reports/eda_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from datetime import date
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

RANGES = {
    "sleep": (0, 14),
    "stress": (1, 10),
    "mood": (1, 10),
    "energy": (1, 10),
    "screenTime": (0, 24),
    "workHours": (0, 24),
    "activity": (0, 300),
    "social": (1, 10),
}


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


def parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path):
    rows = []
    missing = Counter()
    invalid = Counter()
    out_of_range = Counter()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            parsed_date = parse_date(raw.get("date", ""))
            if parsed_date is None:
                invalid["date"] += 1
                continue

            row = {
                "student_id": raw.get("student_id", "unknown") or "unknown",
                "date": parsed_date.isoformat(),
            }
            valid = True
            for feature in FEATURES:
                value = raw.get(feature)
                if value in (None, ""):
                    missing[feature] += 1
                    valid = False
                    continue
                try:
                    number = float(value)
                except ValueError:
                    invalid[feature] += 1
                    valid = False
                    continue
                low, high = RANGES[feature]
                if not low <= number <= high:
                    out_of_range[feature] += 1
                row[feature] = number
            if valid:
                rows.append(row)

    return rows, missing, invalid, out_of_range


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0 or y_denom == 0:
        return 0
    return numerator / (x_denom * y_denom)


def numeric_summary(rows: list[dict]) -> dict:
    summary = {}
    for feature in FEATURES:
        values = [row[feature] for row in rows]
        summary[feature] = {
            "min": round(min(values), 3),
            "q1": round(percentile(values, 0.25), 3),
            "mean": round(mean(values), 3),
            "median": round(percentile(values, 0.5), 3),
            "q3": round(percentile(values, 0.75), 3),
            "max": round(max(values), 3),
            "std": round(stdev(values), 3),
        }
    return summary


def score_summary(rows: list[dict]) -> dict:
    scores = [wellbeing_score(row) for row in rows]
    domain_rows = [domain_scores(row) for row in rows]
    domains = {}
    for domain in ["physical", "mental", "social", "occupational", "digital"]:
        values = [row[domain] for row in domain_rows]
        domains[domain] = {
            "mean": round(mean(values), 3),
            "min": min(values),
            "max": max(values),
        }
    return {
        "wellbeing": {
            "mean": round(mean(scores), 3),
            "min": min(scores),
            "max": max(scores),
            "risk_count_score_under_70": sum(1 for score in scores if score < 70),
        },
        "domains": domains,
    }


def feature_correlations(rows: list[dict]) -> dict:
    scores = [wellbeing_score(row) for row in rows]
    return {feature: round(corr([row[feature] for row in rows], scores), 3) for feature in FEATURES}


def write_report(path: Path, summary: dict):
    lines = [
        "# EDA Report - Personal Wellbeing Time-Series Monitoring App",
        "",
        "## Dataset Overview",
        "",
        f"- Rows: {summary['rows']}",
        f"- Students: {summary['students']}",
        f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}",
        f"- Risk rows, wellbeing score < 70: {summary['scores']['wellbeing']['risk_count_score_under_70']}",
        "",
        "## Preprocessing Checks",
        "",
        f"- Missing values: `{summary['quality']['missing']}`",
        f"- Invalid values: `{summary['quality']['invalid']}`",
        f"- Out-of-range values: `{summary['quality']['out_of_range']}`",
        "",
        "## Numeric Summary",
        "",
        "| Feature | Min | Q1 | Mean | Median | Q3 | Max | Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for feature, values in summary["numeric"].items():
        lines.append(
            f"| {feature} | {values['min']} | {values['q1']} | {values['mean']} | "
            f"{values['median']} | {values['q3']} | {values['max']} | {values['std']} |"
        )

    lines += [
        "",
        "## Wellbeing and Domain Scores",
        "",
        f"- Overall wellbeing mean: {summary['scores']['wellbeing']['mean']}",
        f"- Overall wellbeing min/max: {summary['scores']['wellbeing']['min']} / {summary['scores']['wellbeing']['max']}",
        "",
        "| Domain | Mean | Min | Max |",
        "| --- | ---: | ---: | ---: |",
    ]

    for domain, values in summary["scores"]["domains"].items():
        lines.append(f"| {domain} | {values['mean']} | {values['min']} | {values['max']} |")

    lines += [
        "",
        "## Feature Correlation With Wellbeing Score",
        "",
        "| Feature | Correlation |",
        "| --- | ---: |",
    ]

    for feature, value in summary["correlations_with_wellbeing"].items():
        lines.append(f"| {feature} | {value} |")

    lines += [
        "",
        "## Notes",
        "",
        "- This EDA uses the processed app-ready CSV, not the raw StudentLife files.",
        "- The current score is a research prototype score, not a clinical measure.",
        "- Correlations are exploratory and should not be interpreted as causal.",
        "",
    ]

    path.write_text("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 scripts/eda_processed_data.py data/studentlife_wellbeing.csv")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else Path("reports").resolve()
    rows, missing, invalid, out_of_range = load_rows(input_path)
    if not rows:
        raise SystemExit(f"No valid rows found in {input_path}")

    dates = sorted(row["date"] for row in rows)
    students = sorted({row["student_id"] for row in rows})
    summary = {
        "source": str(input_path),
        "rows": len(rows),
        "students": len(students),
        "student_ids": students,
        "date_range": {"start": dates[0], "end": dates[-1]},
        "quality": {
            "missing": dict(missing),
            "invalid": dict(invalid),
            "out_of_range": dict(out_of_range),
        },
        "numeric": numeric_summary(rows),
        "scores": score_summary(rows),
        "correlations_with_wellbeing": feature_correlations(rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(output_dir / "eda_report.md", summary)

    print(f"Rows: {summary['rows']}")
    print(f"Students: {summary['students']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Wrote {output_dir / 'eda_report.md'}")
    print(f"Wrote {output_dir / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
