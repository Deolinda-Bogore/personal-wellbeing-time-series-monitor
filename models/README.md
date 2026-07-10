# Models

This folder contains two model training pipelines.

## Baseline Model

The baseline model is intentionally simple: a standard-library logistic regression model that predicts whether an entry is at risk based on the same wellbeing score used by the app.

Input CSV:

```text
student_id,date,sleep,stress,mood,energy,screenTime,workHours,activity,social
```

Train:

```bash
python3 models/train_baseline.py data/studentlife_wellbeing.csv
```

Outputs:

```text
models/artifacts/baseline_risk_model.json
models/artifacts/baseline_metrics.json
```

This baseline is a first research step. It is useful because it runs without external ML dependencies and gives a quick sanity check.

## Stronger Time-Series Model

The stronger model uses scikit-learn and trains a Random Forest classifier to predict **next-day wellbeing risk**:

```text
next_day_risk = next_day_wellbeing_score < 70
```

This is a more realistic task than the baseline because the model predicts tomorrow's risk using today's signals, lag features, rolling averages, and domain scores.

Train:

```bash
python3 models/train_wellbeing_model.py data/studentlife_wellbeing.csv
```

Outputs:

```text
models/artifacts/wellbeing_risk_model.joblib
models/artifacts/wellbeing_risk_metrics.json
models/artifacts/wellbeing_feature_importance.csv
```

Current stronger-model result:

```text
Accuracy: 0.786
Precision: 0.814
Recall: 0.717
F1: 0.763
ROC-AUC: 0.846
```

The train/test split holds out students, not just random rows, so the evaluation better tests whether the model generalizes across people.
