# Personal Wellbeing Time-Series Monitoring App

A digital-health research prototype for tracking personal wellbeing signals over time, calculating domain-level scores, detecting early risk patterns, and preparing data for machine learning experiments.

This project explores how machine learning, smartphone and wearable-style sensing, self-reports, time-series modeling, and context-aware analysis can support personal wellbeing monitoring.

## Project Goal

Most digital health tools track a single signal, such as sleep, steps, or stress. This prototype explores a broader idea: daily wellbeing should be understood across multiple connected domains.

The app tracks signals related to:

- physical wellbeing
- mental wellbeing
- social wellbeing
- occupational or academic load
- digital behavior

It then uses those signals to produce a wellbeing score, domain scores, a time-series chart, and explainable risk feedback.

## Research Direction

This prototype focuses on fusing multimodal wellbeing signals across interconnected life domains. It demonstrates:

1. **Daily data collection** through self-report and phone-like signals.
2. **Domain-level scoring** across physical, mental, social, occupational, and digital wellbeing.
3. **Time-series monitoring** instead of one-time wellbeing assessment.
4. **Explainable risk feedback** showing why wellbeing may be declining.
5. **Public dataset preparation** using the Dartmouth StudentLife dataset.
6. **EDA and data quality checks** before modeling.
7. **Baseline and stronger model training** for wellbeing-risk prediction.

## Current Features

- Add or update a daily wellbeing entry.
- Generate sample data for a quick 21-day demo.
- Import processed public-data CSV files.
- Run one combined Streamlit research dashboard.
- View overall wellbeing score and risk status.
- View physical, mental, social, occupational, and digital domain scores.
- Visualize wellbeing trends over time.
- Explain which factors may be contributing to risk.
- Use a dedicated prediction page to enter wellbeing signals and predict next-day risk.
- Display Random Forest next-day risk predictions in the dashboard.
- Optionally serve model predictions through a FastAPI backend.
- Export entries and calculated scores as CSV.
- Prepare StudentLife data for the app.
- Generate an EDA report for the processed dataset.
- Train a baseline wellbeing-risk model.
- Train a stronger next-day risk model with lag features and rolling averages.

## App Fields

The app uses this student-level daily CSV format:

```text
student_id,date,sleep,stress,mood,energy,screenTime,workHours,activity,social
```

Example:

```text
u00,2013-03-27,8.0,1,6.1,8.1,6.1,5.5,17.6,4.9
```

## How to Run Everything

Start from the project folder:

```bash
cd /Users/macbookpro2020m1/personal-wellbeing-time-series-monitor
```

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### 1. Run the Streamlit App

The main application is the combined Streamlit dashboard:

```bash
python3 -m streamlit run streamlit_app.py
```

The app includes:

- **Research Dashboard** for StudentLife time-series analysis
- **Prediction** for filling one wellbeing entry and predicting next-day risk
- **Daily Entry** for saving manual wellbeing entries
- **Data and Export** for reviewing and downloading processed data

If the FastAPI URL field in the sidebar is empty, Streamlit uses the saved local Random Forest model:

```text
models/artifacts/wellbeing_risk_model.joblib
```

### 2. Run the FastAPI Backend

```bash
python3 -m uvicorn api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

Then run Streamlit in another terminal:

```bash
python3 -m streamlit run streamlit_app.py
```

In the Streamlit sidebar, enter:

```text
http://127.0.0.1:8000
```

That connects the Streamlit prediction page to the FastAPI `/predict` endpoint.

### 3. Run EDA

Basic EDA:

```bash
python3 scripts/eda_processed_data.py data/studentlife_wellbeing.csv
```

Enhanced EDA with charts:

```bash
python3 scripts/enhanced_eda.py data/studentlife_wellbeing.csv
```

Outputs are saved in:

```text
reports/
reports/figures/
```

### 4. Train the Models

Train the baseline model:

```bash
python3 models/train_baseline.py data/studentlife_wellbeing.csv
```

Train the stronger Random Forest model:

```bash
python3 models/train_wellbeing_model.py data/studentlife_wellbeing.csv
```

Compare Logistic Regression and Random Forest:

```bash
python3 models/compare_models.py data/studentlife_wellbeing.csv
```

Model artifacts are saved in:

```text
models/artifacts/
```

### 5. Open the Research Notebook

Open locally:

```bash
jupyter notebook notebooks/personal_wellbeing_studentlife_pipeline.ipynb
```

Or open in Google Colab:

[Open notebook in Colab](https://colab.research.google.com/github/Deolinda-Bogore/personal-wellbeing-time-series-monitor/blob/main/notebooks/personal_wellbeing_studentlife_pipeline.ipynb)

The notebook loads the processed CSV from this repository, so no `dataset.zip` upload is required in Colab.

## Streamlit App Details

The dashboard combines data entry, prediction, analysis, and export in one interface. It lets you:

- select a StudentLife participant
- make a new next-day risk prediction from form inputs
- add manual daily wellbeing entries
- load 21-day demo entries
- upload processed CSV files
- view wellbeing and domain timelines
- inspect rule-based burnout-risk flags
- view the deployed Random Forest next-day risk prediction
- connect to an optional FastAPI prediction backend
- read explainable feedback for the latest entry
- review correlations between wellbeing domains
- download the combined processed time-series CSV

## Optional FastAPI Backend

The Streamlit app can load the model locally, but the project also includes a FastAPI backend for a more production-style architecture:

```text
Streamlit dashboard -> FastAPI /predict endpoint -> Random Forest model artifact
```

Run the API:

```bash
python3 -m uvicorn api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Then run Streamlit and enter this URL in the sidebar field named **FastAPI URL**:

```text
http://127.0.0.1:8000
```

You can also set it with an environment variable:

```bash
export WELLBEING_API_URL=http://127.0.0.1:8000
python3 -m streamlit run streamlit_app.py
```

## Research Notebook

The project includes a Colab-friendly research notebook that uses the processed CSV already stored in this repository:

```text
notebooks/personal_wellbeing_studentlife_pipeline.ipynb
```

Open it in Google Colab:

[Open notebook in Colab](https://colab.research.google.com/github/Deolinda-Bogore/personal-wellbeing-time-series-monitor/blob/main/notebooks/personal_wellbeing_studentlife_pipeline.ipynb)

The notebook covers:

- loading the processed StudentLife daily CSV directly from GitHub
- schema mapping from daily features to five wellbeing domains
- domain scoring and normalization
- rule-based burnout risk detection
- lag and rolling-window feature engineering
- next-day risk modeling with a student-level train/test split
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

In Google Colab, no `dataset.zip` upload is required for this notebook. It loads `data/studentlife_wellbeing.csv` from the GitHub repository.

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

Then open the app and use **Upload processed CSV** in the sidebar if you want to load that file manually.

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

For a richer visual EDA report:

```bash
python3 scripts/enhanced_eda.py data/studentlife_wellbeing.csv
```

This creates:

```text
reports/enhanced_eda_report.md
reports/enhanced_eda_summary.json
reports/figures/
```

The enhanced EDA includes wellbeing-score distribution, domain-score correlations, raw feature distributions, student-level risk-day rates, and sample student timelines.

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

To train the stronger scikit-learn model:

```bash
python3 models/train_wellbeing_model.py data/studentlife_wellbeing.csv
```

This model predicts next-day wellbeing risk using current-day signals, domain scores, lag features, and rolling averages. It uses a student-level train/test split so the test set contains held-out students.

The stronger model creates:

```text
models/artifacts/wellbeing_risk_model.joblib
models/artifacts/wellbeing_risk_metrics.json
models/artifacts/wellbeing_feature_importance.csv
```

To compare Logistic Regression and Random Forest on the same next-day prediction task:

```bash
python3 models/compare_models.py data/studentlife_wellbeing.csv
```

This creates:

```text
models/artifacts/model_comparison.csv
models/artifacts/model_comparison.json
```

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

The stronger Random Forest next-day risk model produced:

```text
Accuracy: 0.786
Precision: 0.814
Recall: 0.717
F1: 0.763
ROC-AUC: 0.846
```

The fair next-day model comparison produced:

```text
Model                 Accuracy  Precision  Recall  F1     ROC-AUC
Random Forest          0.786      0.814    0.717  0.763   0.846
Logistic Regression    0.778      0.827    0.677  0.745   0.844
```

These metrics are for a prototype risk label derived from the project wellbeing score, not a clinical label. The stronger model is harder and more realistic than the baseline because it predicts next-day risk for held-out students.

## Training Pipeline

The full data and training pipeline:

1. Downloads the public StudentLife dataset.
2. Preprocesses raw sensing/self-report data into an app-ready student-level daily CSV.
3. Runs EDA and data quality checks on the processed CSV.
4. Uses sleep, stress, mood, energy, screen time, work hours, activity, and social connection as model features.
5. Calculates domain scores and an overall wellbeing score.
6. Creates a binary risk label.
7. Trains a baseline logistic regression model.
8. Trains a stronger Random Forest next-day risk model with lag and rolling-window features.
9. Compares models on the same next-day prediction task.
10. Saves trained model artifacts, comparison results, feature importance, and evaluation metrics.

## Future ML Improvements

The next research steps are:

- Add Gradient Boosting, XGBoost, or LightGBM for stronger tabular modeling.
- Add separate models for each wellbeing domain.
- Compare rule-based risk scoring against learned model predictions.
- Build a true sequence model using LSTM, GRU, Temporal CNN, or Transformer architecture.
- Improve explainability using feature importance or SHAP-style explanations.

## Suggested CV Description

```text
Personal Wellbeing Time-Series Monitoring App - Independent Research Prototype
- Built a digital health prototype for tracking student-level daily wellbeing signals across sleep, stress, mood, energy, screen time, workload, physical activity, and social connection.
- Implemented domain-level scoring across physical, mental, social, occupational, and digital wellbeing, with time-series visualization and explainable risk feedback.
- Added CSV import/export, StudentLife dataset preparation, EDA reporting, and model training pipelines for wellbeing-risk prediction.
- Trained and compared Logistic Regression and Random Forest classifiers for next-day wellbeing-risk prediction, then served the best model through a Streamlit interface and optional FastAPI backend.
```

## Repository Structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── README.md
├── notebooks/
│   └── personal_wellbeing_studentlife_pipeline.ipynb
├── data/
│   ├── README.md
│   └── studentlife_wellbeing.csv
├── scripts/
│   ├── eda_processed_data.py
│   ├── enhanced_eda.py
│   └── prepare_studentlife.py
├── models/
│   ├── README.md
│   ├── compare_models.py
│   ├── train_baseline.py
│   ├── train_wellbeing_model.py
│   └── artifacts/
├── reports/
│   ├── eda_report.md
│   ├── enhanced_eda_report.md
│   └── figures/
└── README.md
```

## Note

This is a research prototype, not a medical diagnostic tool. The risk score is designed for exploration and project demonstration, not clinical decision-making.
