# Models

This folder contains the baseline model training pipeline.

The first model is intentionally simple: a standard-library logistic regression model that predicts whether an entry is at risk based on the same wellbeing score used by the app.

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

This baseline is a first research step. A future version should replace it with a stronger time-series model such as Random Forest with lag features, LSTM/GRU, Temporal CNN, or Transformer-based sequence modeling.
