from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "data" / "studentlife_wellbeing.csv"


st.set_page_config(
    page_title="Personal Wellbeing Time-Series Monitor",
    page_icon=":bar_chart:",
    layout="wide",
)


def clamp(series, lower=0, upper=100):
    return series.clip(lower=lower, upper=upper)


@st.cache_data
def load_default_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    required = [
        "student_id",
        "date",
        "sleep",
        "stress",
        "mood",
        "energy",
        "screenTime",
        "workHours",
        "activity",
        "social",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()

    clean = df[required].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    for column in required[2:]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["student_id", "date"]).sort_values(["student_id", "date"])
    return clean


def add_scores(df):
    scored = df.copy()
    sleep_score = clamp((scored["sleep"] / 8) * 100)
    stress_score = clamp(100 - (scored["stress"] - 1) * 11.1)
    mood_score = clamp(scored["mood"] * 10)
    energy_score = clamp(scored["energy"] * 10)
    social_score = clamp(scored["social"] * 10)
    activity_score = clamp((scored["activity"] / 45) * 100)
    screen_score = 100 - clamp((scored["screenTime"] - 5).clip(lower=0) * 10, 0, 60)
    work_score = 100 - clamp((scored["workHours"] - 8).clip(lower=0) * 12, 0, 60)

    scored["physical_score"] = (sleep_score * 0.45 + activity_score * 0.30 + energy_score * 0.25).round(1)
    scored["mental_score"] = (stress_score * 0.50 + mood_score * 0.35 + energy_score * 0.15).round(1)
    scored["social_score"] = social_score.round(1)
    scored["occupational_score"] = (work_score * 0.55 + stress_score * 0.30 + energy_score * 0.15).round(1)
    scored["digital_score"] = (screen_score * 0.70 + sleep_score * 0.15 + stress_score * 0.15).round(1)

    scored["wellbeing_score"] = (
        scored["physical_score"] * 0.24
        + scored["mental_score"] * 0.28
        + scored["social_score"] * 0.14
        + scored["occupational_score"] * 0.17
        + scored["digital_score"] * 0.17
    ).round(1)
    scored["status"] = pd.cut(
        scored["wellbeing_score"],
        bins=[-1, 49.999, 69.999, 100],
        labels=["High risk", "Watch", "Stable"],
    ).astype(str)
    return scored


def add_rule_engine(df):
    scored = df.copy()
    scored["occupational_strain"] = 100 - scored["occupational_score"]
    for column in ["physical_score", "social_score", "occupational_strain"]:
        grouped = scored.groupby("student_id", group_keys=False)
        scored[f"{column}_rolling_3d"] = grouped[column].apply(
            lambda values: values.rolling(3, min_periods=2).mean()
        )
        scored[f"{column}_previous_3d"] = scored.groupby("student_id")[f"{column}_rolling_3d"].shift(3)

    strain_base = scored["occupational_strain_previous_3d"].replace(0, 1)
    physical_base = scored["physical_score_previous_3d"].replace(0, 1)
    social_base = scored["social_score_previous_3d"].replace(0, 1)

    scored["strain_change"] = (scored["occupational_strain_rolling_3d"] - strain_base) / strain_base
    scored["physical_drop"] = (physical_base - scored["physical_score_rolling_3d"]) / physical_base
    scored["social_drop"] = (social_base - scored["social_score_rolling_3d"]) / social_base

    scored["burnout_risk"] = (
        (scored["strain_change"] > 0.30)
        & (scored["physical_drop"] > 0.20)
        & (scored["social_drop"] > 0.20)
    )
    return scored


def explain_latest(row):
    reasons = []

    if row["burnout_risk"]:
        reasons.append(
            "Rule-based burnout risk triggered: academic strain rose while physical and social wellbeing dropped over the recent rolling window."
        )
    if row["sleep"] < 6:
        reasons.append("Sleep is below 6 hours, lowering the physical recovery signal.")
    if row["stress"] >= 8:
        reasons.append("Stress is high, increasing the mental and occupational risk signals.")
    if row["screenTime"] > 8:
        reasons.append("Screen time is high, increasing digital-load risk.")
    if row["workHours"] > 9:
        reasons.append("Academic or work load is high, which may contribute to burnout risk.")
    if row["social"] <= 4:
        reasons.append("Social connection is low, suggesting possible social withdrawal.")

    if not reasons:
        reasons.append("No major risk driver dominates the latest entry; domain signals are relatively balanced.")
    return reasons


raw_df = load_default_data()

st.title("Personal Wellbeing Time-Series Monitor")
st.caption("KAUST-aligned research prototype using StudentLife daily sensing and self-report signals.")

uploaded_file = st.sidebar.file_uploader("Upload processed CSV", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

df = add_rule_engine(add_scores(clean_data(raw_df))) if not raw_df.empty else pd.DataFrame()

if df.empty:
    st.info("Add `data/studentlife_wellbeing.csv` or upload a processed CSV to begin.")
    st.stop()

students = sorted(df["student_id"].dropna().unique())
selected_student = st.sidebar.selectbox("Student", students)
student_df = df[df["student_id"] == selected_student].sort_values("date").copy()
latest = student_df.iloc[-1]

score_delta = None
if len(student_df) > 1:
    score_delta = latest["wellbeing_score"] - student_df.iloc[-2]["wellbeing_score"]

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Latest wellbeing", f"{latest['wellbeing_score']:.1f}", None if score_delta is None else f"{score_delta:+.1f}")
metric_2.metric("Risk status", latest["status"])
metric_3.metric("Entries", len(student_df))
metric_4.metric("Burnout flags", int(student_df["burnout_risk"].sum()))

domain_columns = [
    "physical_score",
    "mental_score",
    "social_score",
    "occupational_score",
    "digital_score",
]

st.subheader("Domain Timeline")
timeline_df = student_df.set_index("date")[domain_columns + ["wellbeing_score"]]
st.line_chart(timeline_df)

left, right = st.columns([1, 1])

with left:
    st.subheader("Latest Domain Scores")
    latest_domains = latest[domain_columns].rename(
        {
            "physical_score": "Physical",
            "mental_score": "Mental",
            "social_score": "Social",
            "occupational_score": "Occupational",
            "digital_score": "Digital",
        }
    )
    st.bar_chart(latest_domains)

with right:
    st.subheader("Explainable Risk Feedback")
    for reason in explain_latest(latest):
        st.write(f"- {reason}")

st.subheader("Burnout Risk Overlay")
risk_days = student_df[student_df["burnout_risk"]][["date", "wellbeing_score", "strain_change", "physical_drop", "social_drop"]]
if risk_days.empty:
    st.write("No rule-based burnout risk days were detected for this student.")
else:
    st.dataframe(risk_days, use_container_width=True, hide_index=True)

st.subheader("Correlation Snapshot")
corr = student_df[domain_columns + ["wellbeing_score"]].corr().round(2)
st.dataframe(corr, use_container_width=True)

st.subheader("Processed Daily Data")
st.dataframe(student_df, use_container_width=True, hide_index=True)

csv = student_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download selected student CSV",
    data=csv,
    file_name=f"{selected_student}_wellbeing_timeseries.csv",
    mime="text/csv",
)
