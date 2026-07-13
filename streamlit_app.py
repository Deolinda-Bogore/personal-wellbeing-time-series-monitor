import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "data" / "studentlife_wellbeing.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "artifacts" / "wellbeing_risk_model.joblib"
REQUIRED_COLUMNS = [
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
DOMAIN_COLUMNS = [
    "physical_score",
    "mental_score",
    "social_score",
    "occupational_score",
    "digital_score",
]
MODEL_BASE_COLUMNS = REQUIRED_COLUMNS[2:] + DOMAIN_COLUMNS + ["wellbeing_score"]


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
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None
    return joblib.load(MODEL_PATH)


def build_sample_entries():
    start = date.today() - timedelta(days=20)
    rows = []
    for index in range(21):
        rows.append(
            {
                "student_id": "manual_demo",
                "date": start + timedelta(days=index),
                "sleep": round(7.8 - (index % 5) * 0.35, 1),
                "stress": min(10, 3 + (index % 6)),
                "mood": max(1, 8 - (index % 4)),
                "energy": max(1, 8 - (index % 5) * 0.6),
                "screenTime": round(4.5 + (index % 6) * 0.7, 1),
                "workHours": round(5.5 + (index % 7) * 0.6, 1),
                "activity": round(35 - (index % 6) * 3.2, 1),
                "social": max(1, 8 - (index % 5) * 0.7),
            }
        )
    return pd.DataFrame(rows)


def clean_data(df):
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    clean = df[REQUIRED_COLUMNS].copy()
    clean["student_id"] = clean["student_id"].fillna("manual").astype(str)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    for column in REQUIRED_COLUMNS[2:]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=["student_id", "date"])
    clean = clean.dropna(subset=REQUIRED_COLUMNS[2:], how="all")
    clean = clean.sort_values(["student_id", "date"]).reset_index(drop=True)
    return clean


def add_scores(df):
    if df.empty:
        return df

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
    if df.empty:
        return df

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


def add_model_features(df):
    if df.empty:
        return df

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


def predict_next_day_risk(model_artifact, feature_df, selected_student):
    if model_artifact is None or feature_df.empty:
        return None

    metadata = model_artifact.get("metadata", {})
    feature_columns = metadata.get("features", [])
    pipeline = model_artifact.get("pipeline")
    if pipeline is None or not feature_columns:
        return None

    student_rows = feature_df[feature_df["student_id"] == selected_student].sort_values("date")
    if student_rows.empty:
        return None

    latest_features = student_rows.iloc[[-1]][feature_columns]
    probability = float(pipeline.predict_proba(latest_features)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability": probability,
        "label": "Elevated risk" if prediction else "Lower risk",
        "metadata": metadata,
        "source": "local_model",
    }


def predict_next_day_risk_from_api(api_url, student_df):
    if not api_url or student_df.empty:
        return None

    try:
        import requests
    except ImportError:
        return {
            "error": "The requests package is not installed. Run `python3 -m pip install -r requirements.txt`."
        }

    endpoint = api_url.rstrip("/") + "/predict"
    payload = {
        "student_id": str(student_df.iloc[-1]["student_id"]),
        "entries": student_df[REQUIRED_COLUMNS].assign(date=student_df["date"].dt.strftime("%Y-%m-%d")).to_dict(
            orient="records"
        ),
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return {"error": f"FastAPI request failed: {error}"}

    return {
        "prediction": int(data["prediction"]),
        "probability": float(data["probability"]),
        "label": data["label"],
        "metadata": {
            "model_type": data.get("model_type"),
            "task": data.get("task"),
            "metrics": data.get("metrics", {}),
        },
        "source": "fastapi",
    }


def explain_latest(row):
    reasons = []
    lowest_domain = row[DOMAIN_COLUMNS].astype(float).idxmin().replace("_score", "")

    if row["burnout_risk"]:
        reasons.append(
            "Rule-based burnout risk triggered because academic strain rose while physical and social wellbeing dropped."
        )
    if row[lowest_domain + "_score"] < 65:
        reasons.append(f"The lowest current domain is {lowest_domain} wellbeing.")
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


def explain_model_prediction(prediction):
    if prediction is None:
        return [
            "Model artifact is not available yet. Train the Random Forest model or install the project requirements."
        ]
    if prediction.get("error"):
        return [prediction["error"]]

    probability = prediction["probability"]
    source = "FastAPI backend" if prediction.get("source") == "fastapi" else "local model artifact"
    if probability >= 0.7:
        return [
            f"The Random Forest model estimates a high next-day risk probability ({probability:.1%}) using the {source}.",
            "This prediction uses current signals plus lag and rolling-window features from recent days.",
        ]
    if probability >= 0.5:
        return [
            f"The Random Forest model estimates elevated next-day risk ({probability:.1%}) using the {source}.",
            "The result is close enough to the threshold that recent patterns should be reviewed alongside the rule-based explanation.",
        ]
    return [
        f"The Random Forest model estimates lower next-day risk ({probability:.1%}) using the {source}.",
        "This is a prototype prediction based on the processed wellbeing score, not a clinical assessment.",
    ]


def render_prediction_panel(prediction):
    st.subheader("Next-Day Risk Prediction")

    if prediction is None:
        st.info("Model prediction is unavailable. Check that the model artifact exists or start the FastAPI backend.")
        return

    if prediction.get("error"):
        st.error(prediction["error"])
        return

    probability = prediction["probability"]
    source = "FastAPI backend" if prediction.get("source") == "fastapi" else "local model artifact"
    if prediction["prediction"]:
        st.warning(f"Elevated next-day risk predicted: {probability:.1%}")
    else:
        st.success(f"Lower next-day risk predicted: {probability:.1%}")

    st.progress(min(max(probability, 0), 1))
    st.caption(f"Prediction source: {source}. Threshold: 50%. Model: Random Forest classifier.")


def current_dataset():
    frames = []
    if st.session_state.include_studentlife:
        frames.append(load_default_data())
    if st.session_state.uploaded_df is not None:
        frames.append(st.session_state.uploaded_df)
    if not st.session_state.manual_df.empty:
        frames.append(st.session_state.manual_df)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def add_manual_entry(entry):
    new_row = pd.DataFrame([entry])
    st.session_state.manual_df = pd.concat([st.session_state.manual_df, new_row], ignore_index=True)


def predict_custom_entry(entry, api_url, model_artifact):
    context = st.session_state.manual_df.copy()
    if not context.empty:
        context = context[context["student_id"].astype(str) == str(entry["student_id"])]

    prediction_input = pd.concat([context, pd.DataFrame([entry])], ignore_index=True)
    scored_input = add_rule_engine(add_scores(clean_data(prediction_input)))
    feature_input = add_model_features(scored_input)
    selected_student = str(entry["student_id"])

    prediction = (
        predict_next_day_risk_from_api(api_url, scored_input)
        if api_url
        else predict_next_day_risk(model_artifact, feature_input, selected_student)
    )
    return prediction, scored_input


if "manual_df" not in st.session_state:
    st.session_state.manual_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "include_studentlife" not in st.session_state:
    st.session_state.include_studentlife = True
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_scored_input" not in st.session_state:
    st.session_state.prediction_scored_input = pd.DataFrame()
if "prediction_entry" not in st.session_state:
    st.session_state.prediction_entry = None


st.title("Personal Wellbeing Time-Series Monitor")
st.caption("One Streamlit app for data entry, StudentLife analysis, explainable risk feedback, and CSV export.")

with st.sidebar:
    st.header("Data")
    st.checkbox("Include StudentLife processed data", key="include_studentlife")
    api_url = st.text_input(
        "FastAPI URL",
        value=os.environ.get("WELLBEING_API_URL", ""),
        placeholder="http://127.0.0.1:8000",
        help="Optional. If provided, Streamlit will request predictions from FastAPI instead of the local model.",
    )
    uploaded_file = st.file_uploader("Upload processed CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.uploaded_df = pd.read_csv(uploaded_file)
        st.success("Uploaded CSV added.")
    if st.button("Load 21-day manual demo"):
        st.session_state.manual_df = build_sample_entries()
        st.success("Demo entries loaded.")
    if st.button("Clear manual entries"):
        st.session_state.manual_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        st.success("Manual entries cleared.")

raw_df = current_dataset()
df = add_rule_engine(add_scores(clean_data(raw_df)))
model_feature_df = add_model_features(df)
model_artifact = load_model_artifact()

dashboard_tab, prediction_tab, entry_tab, data_tab = st.tabs(
    ["Research Dashboard", "Prediction", "Daily Entry", "Data and Export"]
)

with dashboard_tab:
    if df.empty:
        st.info("Add manual entries, upload a processed CSV, or include the StudentLife processed dataset.")
    else:
        students = sorted(df["student_id"].dropna().unique())
        selected_student = st.selectbox("Student time series", students)
        student_df = df[df["student_id"] == selected_student].sort_values("date").copy()
        latest = student_df.iloc[-1]
        model_prediction = (
            predict_next_day_risk_from_api(api_url, student_df)
            if api_url
            else predict_next_day_risk(model_artifact, model_feature_df, selected_student)
        )

        score_delta = None
        if len(student_df) > 1:
            score_delta = latest["wellbeing_score"] - student_df.iloc[-2]["wellbeing_score"]

        metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
        metric_1.metric(
            "Latest wellbeing",
            f"{latest['wellbeing_score']:.1f}",
            None if score_delta is None else f"{score_delta:+.1f}",
        )
        metric_2.metric("Risk status", latest["status"])
        metric_3.metric("Entries", len(student_df))
        metric_4.metric("Burnout flags", int(student_df["burnout_risk"].sum()))
        if model_prediction is None:
            metric_5.metric("Model next-day risk", "Unavailable")
        elif model_prediction.get("error"):
            metric_5.metric("Model next-day risk", "API error")
        else:
            metric_5.metric("Model next-day risk", model_prediction["label"], f"{model_prediction['probability']:.1%}")

        render_prediction_panel(model_prediction)

        st.subheader("Domain Timeline")
        timeline_df = student_df.set_index("date")[DOMAIN_COLUMNS + ["wellbeing_score"]]
        st.line_chart(timeline_df)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Latest Domain Scores")
            latest_domains = latest[DOMAIN_COLUMNS].rename(
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

            st.subheader("Model Prediction")
            for reason in explain_model_prediction(model_prediction):
                st.write(f"- {reason}")

            if model_prediction is not None and not model_prediction.get("error"):
                metrics = model_prediction["metadata"].get("metrics", {})
                source = "FastAPI" if model_prediction.get("source") == "fastapi" else "local artifact"
                st.caption(
                    f"Prediction source: {source}. Random Forest validation: "
                    f"F1 {metrics.get('f1', 'n/a')}, ROC-AUC {metrics.get('roc_auc', 'n/a')}"
                )

        st.subheader("Burnout Risk Overlay")
        risk_days = student_df[student_df["burnout_risk"]][
            ["date", "wellbeing_score", "strain_change", "physical_drop", "social_drop"]
        ]
        if risk_days.empty:
            st.write("No rule-based burnout risk days were detected for this student.")
        else:
            st.dataframe(risk_days, use_container_width=True, hide_index=True)

        st.subheader("Correlation Snapshot")
        corr = student_df[DOMAIN_COLUMNS + ["wellbeing_score"]].corr().round(2)
        st.dataframe(corr, use_container_width=True)

with prediction_tab:
    st.subheader("Make a Next-Day Risk Prediction")
    with st.form("quick_prediction_form"):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            pred_student_id = st.text_input("Student ID", value="manual_predict", key="pred_student_id")
            pred_date = st.date_input("Entry date", value=date.today(), key="pred_date")
            pred_sleep = st.number_input(
                "Sleep hours",
                min_value=0.0,
                max_value=14.0,
                value=7.0,
                step=0.5,
                key="pred_sleep",
            )
        with col_2:
            pred_stress = st.slider("Stress level", 1, 10, 5, key="pred_stress")
            pred_mood = st.slider("Mood", 1, 10, 7, key="pred_mood")
            pred_energy = st.slider("Energy", 1, 10, 7, key="pred_energy")
        with col_3:
            pred_screen_time = st.number_input(
                "Screen time hours",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                key="pred_screen_time",
            )
            pred_work_hours = st.number_input(
                "Study/work hours",
                min_value=0.0,
                max_value=24.0,
                value=6.0,
                step=0.5,
                key="pred_work_hours",
            )
            pred_activity = st.number_input(
                "Activity minutes",
                min_value=0.0,
                max_value=300.0,
                value=30.0,
                step=5.0,
                key="pred_activity",
            )
            pred_social = st.slider("Social connection", 1, 10, 7, key="pred_social")

        predict_submitted = st.form_submit_button("Predict next-day risk")

    if predict_submitted:
        entry = {
            "student_id": pred_student_id.strip() or "manual_predict",
            "date": pred_date,
            "sleep": pred_sleep,
            "stress": pred_stress,
            "mood": pred_mood,
            "energy": pred_energy,
            "screenTime": pred_screen_time,
            "workHours": pred_work_hours,
            "activity": pred_activity,
            "social": pred_social,
        }
        prediction, scored_input = predict_custom_entry(entry, api_url, model_artifact)
        st.session_state.prediction_result = prediction
        st.session_state.prediction_scored_input = scored_input
        st.session_state.prediction_entry = entry

    prediction = st.session_state.prediction_result
    scored_input = st.session_state.prediction_scored_input
    if prediction is not None and not scored_input.empty:
        latest_prediction_row = scored_input.iloc[-1]
        score_col, status_col, probability_col = st.columns(3)
        score_col.metric("Current wellbeing score", f"{latest_prediction_row['wellbeing_score']:.1f}")
        status_col.metric("Current rule status", latest_prediction_row["status"])
        if prediction.get("error"):
            probability_col.metric("Model probability", "Unavailable")
        else:
            probability_col.metric("Model probability", f"{prediction['probability']:.1%}")

        render_prediction_panel(prediction)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Input Domain Scores")
            latest_domains = latest_prediction_row[DOMAIN_COLUMNS].rename(
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
            st.subheader("Why This Result")
            for reason in explain_latest(latest_prediction_row):
                st.write(f"- {reason}")
            for reason in explain_model_prediction(prediction):
                st.write(f"- {reason}")

        if st.session_state.prediction_entry is not None:
            if st.button("Save this entry to Daily Entry data"):
                add_manual_entry(st.session_state.prediction_entry)
                st.success("Prediction entry saved.")

with entry_tab:
    st.subheader("Record a Daily Wellbeing Entry")
    with st.form("manual_entry_form"):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            student_id = st.text_input("Student ID", value="manual")
            entry_date = st.date_input("Date", value=date.today())
            sleep = st.number_input("Sleep hours", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
        with col_2:
            stress = st.slider("Stress level", 1, 10, 5)
            mood = st.slider("Mood", 1, 10, 7)
            energy = st.slider("Energy", 1, 10, 7)
        with col_3:
            screen_time = st.number_input("Screen time hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
            work_hours = st.number_input("Study/work hours", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            activity = st.number_input("Activity minutes", min_value=0.0, max_value=240.0, value=30.0, step=5.0)
            social = st.slider("Social connection", 1, 10, 7)

        submitted = st.form_submit_button("Add entry")
        if submitted:
            add_manual_entry(
                {
                    "student_id": student_id.strip() or "manual",
                    "date": entry_date,
                    "sleep": sleep,
                    "stress": stress,
                    "mood": mood,
                    "energy": energy,
                    "screenTime": screen_time,
                    "workHours": work_hours,
                    "activity": activity,
                    "social": social,
                }
            )
            st.success("Entry added. Open the Research Dashboard tab to see the updated score.")

    st.subheader("Manual Entries")
    if st.session_state.manual_df.empty:
        st.write("No manual entries yet.")
    else:
        st.dataframe(st.session_state.manual_df, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Processed Dataset")
    if df.empty:
        st.write("No data available yet.")
    else:
        st.write(f"{len(df):,} rows across {df['student_id'].nunique():,} student time series.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined processed CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="combined_wellbeing_timeseries.csv",
            mime="text/csv",
        )

    st.subheader("Expected CSV Format")
    st.code(",".join(REQUIRED_COLUMNS), language="text")
