# Personal Wellbeing Time-Series Monitoring App

A digital-health research prototype for tracking personal wellbeing signals over time, calculating domain-level scores, detecting early risk patterns, and preparing data for machine learning experiments.

This project was built as preparation for KAUST's **Wholebeing Digital Twin** research direction, which combines machine learning, smartphone and wearable sensing, self-reports, time-series modeling, and context-aware wellbeing analysis.

## Project Goal

Most digital health tools track a single signal, such as sleep, steps, or stress. This prototype explores a broader idea: daily wellbeing should be understood across multiple connected domains.

The app tracks signals related to:

- physical wellbeing
- mental wellbeing
- social wellbeing
- occupational or academic load
- digital behavior

It then uses those signals to produce a wellbeing score, domain scores, a time-series chart, and explainable risk feedback.

## Why This Matches the KAUST Project

The KAUST **Wholebeing Digital Twin** project focuses on fusing multimodal data from wearables, smartphones, and short self-reports across interconnected life domains.

This prototype aligns with that project because it demonstrates:

1. **Daily data collection** through self-report and phone-like signals.
2. **Domain-level scoring** across physical, mental, social, occupational, and digital wellbeing.
3. **Time-series monitoring** instead of one-time wellbeing assessment.
4. **Explainable risk feedback** showing why wellbeing may be declining.
5. **Public dataset preparation** using the Dartmouth StudentLife dataset.
6. **EDA and data quality checks** before modeling.
7. **Baseline model training** for wellbeing-risk prediction.

## Current Features

- Add or update a daily wellbeing entry.
- Generate sample data for a quick 21-day demo.
- Import processed public-data CSV files.
- Run one combined Streamlit research dashboard.
- View overall wellbeing score and risk status.
- View physical, mental, social, occupational, and digital domain scores.
- Visualize wellbeing trends over time.
- Explain which factors may be contributing to risk.
- Export entries and calculated scores as CSV.
- Prepare StudentLife data for the app.
- Generate an EDA report for the processed dataset.
- Train a baseline wellbeing-risk model.

## App Fields

The app uses this student-level daily CSV format:

```text
student_id,date,sleep,stress,mood,energy,screenTime,workHours,activity,social
```

Example:

```text
u00,2013-03-27,8.0,1,6.1,8.1,6.1,5.5,17.6,4.9
```

## How to Run the App

The main application is the combined Streamlit dashboard:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

The dashboard combines the original browser-app functionality with the research interface. It lets you:

- select a StudentLife participant
- add manual daily wellbeing entries
- load 21-day demo entries
- upload processed CSV files
- view wellbeing and domain timelines
- inspect rule-based burnout-risk flags
- read explainable feedback for the latest entry
- review correlations between wellbeing domains
- download the combined processed time-series CSV

## Optional Browser Prototype

The repository also keeps a lightweight HTML/CSS/JavaScript prototype:

```text
index.html
app.js
styles.css
```

This version is useful as a simple frontend demo, but the Streamlit app is the recommended version for the KAUST research application.

## Research Notebook

The project includes a recruiter-ready notebook that walks through the full research workflow:

```text
notebooks/personal_wellbeing_studentlife_pipeline.ipynb
```

The notebook covers:

- raw StudentLife data ingestion
- schema mapping from raw files to five wellbeing domains
- daily alignment of asynchronous EMA and sensing streams
- domain scoring and normalization
- rule-based burnout risk detection
- EDA visualizations
- clean CSV export for future ML experiments

Install notebook dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/personal_wellbeing_studentlife_pipeline.ipynb
```

If your dataset is not in `~/Downloads/dataset.zip`, set:

```bash
export STUDENTLIFE_ZIP=/path/to/dataset.zip
```

## Data Preprocessing

Download the official Dartmouth StudentLife dataset:

https://studentlife.cs.dartmouth.edu/dataset.html

After downloading `dataset.zip`, preprocess it into the app-ready format:

```bash
python3 scripts/prepare_studentlife.py /path/to/dataset.zip
```

This creates:

```text
data/studentlife_wellbeing.csv
```

Then open the app and click **Import CSV**.

The preprocessing script maps raw asynchronous StudentLife records into a compact daily time-series table with one row per student per day. It extracts and normalizes signals such as sleep, stress, mood, social connection, phone-use proxy, academic workload proxy, and activity proxy into the app schema.

## Exploratory Data Analysis

After preprocessing, generate an EDA report:

```bash
python3 scripts/eda_processed_data.py data/studentlife_wellbeing.csv
```

This creates:

```text
reports/eda_report.md
reports/eda_summary.json
```

The EDA report includes:

- row count and date range
- number of students
- missing-value checks
- invalid-value checks
- out-of-range checks
- numeric summaries for each feature
- wellbeing and domain-score summaries
- feature correlations with the wellbeing score

## How to Start Model Training

First, make sure you have an app-ready CSV. The easiest path is:

```bash
python3 scripts/prepare_studentlife.py /path/to/dataset.zip
python3 scripts/eda_processed_data.py data/studentlife_wellbeing.csv
```

Then train the baseline model:

```bash
python3 models/train_baseline.py data/studentlife_wellbeing.csv
```

The training script creates:

```text
models/artifacts/baseline_risk_model.json
models/artifacts/baseline_metrics.json
```

The current baseline model predicts whether a daily entry is at risk, where:

```text
risk = wellbeing_score < 70
```

This is a first machine-learning baseline. It is intentionally simple and uses only Python's standard library so it can run without installing pandas, NumPy, or scikit-learn.

## Current Processed Dataset and Baseline Results

Using the StudentLife dataset, the current preprocessing pipeline produced:

```text
3,412 daily rows across 49 students
Date range: 2013-03-25 to 2013-08-16
```

The current baseline logistic regression model produced:

```text
Accuracy: 0.943
Precision: 0.958
Recall: 0.927
```

These metrics are for a prototype risk label derived from the project wellbeing score, not a clinical label.

## Training Pipeline

The full data and training pipeline:

1. Downloads the public StudentLife dataset.
2. Preprocesses raw sensing/self-report data into an app-ready student-level daily CSV.
3. Runs EDA and data quality checks on the processed CSV.
4. Uses sleep, stress, mood, energy, screen time, work hours, activity, and social connection as model features.
5. Calculates domain scores and an overall wellbeing score.
6. Creates a binary risk label.
7. Trains a logistic regression classifier.
8. Saves the trained model and evaluation metrics.

## Future ML Improvements

The next research steps are:

- Add lag features, such as previous-day stress and 7-day sleep average.
- Train a Random Forest or Gradient Boosting model.
- Add separate models for each wellbeing domain.
- Compare rule-based risk scoring against learned model predictions.
- Build a true sequence model using LSTM, GRU, Temporal CNN, or Transformer architecture.
- Improve explainability using feature importance or SHAP-style explanations.

## Suggested CV Description

```text
Personal Wellbeing Time-Series Monitoring App - Independent Research Prototype
- Built a digital health prototype for tracking student-level daily wellbeing signals across sleep, stress, mood, energy, screen time, workload, physical activity, and social connection.
- Implemented domain-level scoring across physical, mental, social, occupational, and digital wellbeing, with time-series visualization and explainable risk feedback.
- Added CSV import/export, a StudentLife dataset preparation script, and a baseline model training pipeline for wellbeing-risk prediction.
- Added EDA and data-quality reporting for the processed StudentLife student-level wellbeing time-series dataset.
```

## Repository Structure

```text
.
├── index.html
├── styles.css
├── app.js
├── requirements.txt
├── notebooks/
│   └── personal_wellbeing_studentlife_pipeline.ipynb
├── data/
│   └── README.md
├── scripts/
│   ├── eda_processed_data.py
│   └── prepare_studentlife.py
├── models/
│   ├── README.md
│   └── train_baseline.py
└── README.md
```

## Note

This is a research prototype, not a medical diagnostic tool. The risk score is designed for exploration and project demonstration, not clinical decision-making.
