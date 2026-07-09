# Data

This folder is for processed app-ready CSV files.

Recommended workflow:

1. Download the official Dartmouth StudentLife dataset.
2. Run:

```bash
python3 scripts/prepare_studentlife.py /path/to/dataset.zip
```

3. The script will create:

```text
data/studentlife_wellbeing.csv
```

4. Open the app and use **Import CSV** to load the processed data.

The app-ready CSV columns are:

```text
student_id,date,sleep,stress,mood,energy,screenTime,workHours,activity,social
```
