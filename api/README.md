# FastAPI Backend

This optional backend serves the trained Random Forest model through an API.

Run locally:

```bash
python3 -m uvicorn api.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
POST /predict
```

The endpoint expects a list of daily entries for one or more students and returns the latest next-day risk prediction for the selected student.
