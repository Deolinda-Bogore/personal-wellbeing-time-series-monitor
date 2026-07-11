#!/usr/bin/env python3
"""
Generate expanded EDA artifacts for the processed wellbeing CSV.

Usage:
  python3 scripts/enhanced_eda.py data/studentlife_wellbeing.csv

Outputs:
  reports/enhanced_eda_report.md
  reports/enhanced_eda_summary.json
  reports/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def clamp(series, lower=0, upper=100):
    return series.clip(lower=lower, upper=upper)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
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
    scored["risk_label"] = scored["wellbeing_score"] < 70
    return scored


def load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["student_id", "date"] + RAW_FEATURES).sort_values(["student_id", "date"])


def save_wellbeing_distribution(df: pd.DataFrame, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["wellbeing_score"], bins=30, kde=True, ax=ax, color="#2c7fb8")
    ax.axvline(70, color="#d7301f", linestyle="--", label="Risk threshold")
    ax.set_title("Distribution of Daily Wellbeing Scores")
    ax.set_xlabel("Wellbeing score")
    ax.set_ylabel("Daily rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "wellbeing_score_distribution.png", dpi=160)
    plt.close(fig)


def save_domain_heatmap(df: pd.DataFrame, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[DOMAIN_COLUMNS + ["wellbeing_score"]].corr()
    sns.heatmap(corr, annot=True, cmap="BrBG", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Between Domain Scores")
    fig.tight_layout()
    fig.savefig(figures_dir / "domain_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def save_feature_distributions(df: pd.DataFrame, figures_dir: Path):
    long_df = df[RAW_FEATURES].melt(var_name="feature", value_name="value")
    grid = sns.FacetGrid(long_df, col="feature", col_wrap=4, sharex=False, sharey=False, height=2.5)
    grid.map_dataframe(sns.histplot, x="value", bins=24, color="#41ab5d")
    grid.set_titles("{col_name}")
    grid.fig.suptitle("Raw Feature Distributions", y=1.03)
    grid.fig.tight_layout()
    grid.fig.savefig(figures_dir / "raw_feature_distributions.png", dpi=160)
    plt.close(grid.fig)


def save_risk_rate_by_student(df: pd.DataFrame, figures_dir: Path):
    risk_by_student = (
        df.groupby("student_id")
        .agg(days=("date", "count"), risk_rate=("risk_label", "mean"), mean_wellbeing=("wellbeing_score", "mean"))
        .reset_index()
        .sort_values("risk_rate", ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=risk_by_student, x="student_id", y="risk_rate", ax=ax, color="#756bb1")
    ax.set_title("Top 20 Students by Risk-Day Rate")
    ax.set_xlabel("Student ID")
    ax.set_ylabel("Risk-day rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_rate_by_student.png", dpi=160)
    plt.close(fig)
    return risk_by_student


def save_sample_timelines(df: pd.DataFrame, figures_dir: Path):
    counts = df.groupby("student_id")["date"].count().sort_values(ascending=False)
    sample_students = counts.head(4).index.tolist()
    sample_df = df[df["student_id"].isin(sample_students)]

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.lineplot(data=sample_df, x="date", y="wellbeing_score", hue="student_id", ax=ax)
    ax.axhline(70, color="#d7301f", linestyle="--", label="Risk threshold")
    ax.set_title("Sample Student Wellbeing Timelines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wellbeing score")
    fig.tight_layout()
    fig.savefig(figures_dir / "sample_student_timelines.png", dpi=160)
    plt.close(fig)


def build_summary(df: pd.DataFrame, risk_by_student: pd.DataFrame) -> dict:
    domain_means = df[DOMAIN_COLUMNS].mean().round(3).to_dict()
    raw_summary = df[RAW_FEATURES].describe().round(3).to_dict()
    return {
        "rows": int(len(df)),
        "students": int(df["student_id"].nunique()),
        "date_range": {
            "start": df["date"].min().date().isoformat(),
            "end": df["date"].max().date().isoformat(),
        },
        "risk_rows": int(df["risk_label"].sum()),
        "risk_rate": round(float(df["risk_label"].mean()), 3),
        "mean_wellbeing": round(float(df["wellbeing_score"].mean()), 3),
        "median_wellbeing": round(float(df["wellbeing_score"].median()), 3),
        "domain_means": domain_means,
        "raw_feature_summary": raw_summary,
        "top_risk_students": risk_by_student.round(3).to_dict(orient="records"),
    }


def write_report(path: Path, summary: dict):
    lines = [
        "# Enhanced EDA Report",
        "",
        "## Dataset Overview",
        "",
        f"- Rows: {summary['rows']}",
        f"- Students: {summary['students']}",
        f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}",
        f"- Risk rows, wellbeing score < 70: {summary['risk_rows']}",
        f"- Risk-day rate: {summary['risk_rate']}",
        f"- Mean wellbeing score: {summary['mean_wellbeing']}",
        f"- Median wellbeing score: {summary['median_wellbeing']}",
        "",
        "## Domain Means",
        "",
        "| Domain | Mean score |",
        "| --- | ---: |",
    ]

    for domain, value in summary["domain_means"].items():
        lines.append(f"| {domain} | {value} |")

    lines += [
        "",
        "## Top Students by Risk-Day Rate",
        "",
        "| Student | Days | Risk rate | Mean wellbeing |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in summary["top_risk_students"]:
        lines.append(
            f"| {row['student_id']} | {int(row['days'])} | {row['risk_rate']} | {row['mean_wellbeing']} |"
        )

    lines += [
        "",
        "## Generated Figures",
        "",
        "- `reports/figures/wellbeing_score_distribution.png`",
        "- `reports/figures/domain_correlation_heatmap.png`",
        "- `reports/figures/raw_feature_distributions.png`",
        "- `reports/figures/risk_rate_by_student.png`",
        "- `reports/figures/sample_student_timelines.png`",
        "",
    ]

    path.write_text("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 scripts/enhanced_eda.py data/studentlife_wellbeing.csv")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = add_scores(load_processed(input_path))
    save_wellbeing_distribution(df, figures_dir)
    save_domain_heatmap(df, figures_dir)
    save_feature_distributions(df, figures_dir)
    risk_by_student = save_risk_rate_by_student(df, figures_dir)
    save_sample_timelines(df, figures_dir)

    summary = build_summary(df, risk_by_student)
    (reports_dir / "enhanced_eda_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(reports_dir / "enhanced_eda_report.md", summary)

    print(f"Rows: {summary['rows']}")
    print(f"Students: {summary['students']}")
    print(f"Saved report: {reports_dir / 'enhanced_eda_report.md'}")
    print(f"Saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
