#!/usr/bin/env python3
"""
Prepare app-ready student-level daily rows from the Dartmouth StudentLife dataset.

Usage:
  python3 scripts/prepare_studentlife.py /path/to/dataset.zip
  python3 scripts/prepare_studentlife.py /path/to/extracted/dataset

Output:
  data/studentlife_wellbeing.csv

The script uses only Python's standard library. It maps raw asynchronous
StudentLife EMA and sensing streams into one daily row per student.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


APP_FIELDS = [
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


UID_RE = re.compile(r"_u(\d+)\.")


class DatasetReader:
    def __init__(self, source: Path):
        self.source = source
        self.zip = zipfile.ZipFile(source) if source.is_file() else None
        self._names = self.zip.namelist() if self.zip else [
            str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()
        ]

    def names(self):
        return self._names

    def read_text(self, name: str) -> str:
        if self.zip:
            with self.zip.open(name) as handle:
                return handle.read().decode("utf-8", errors="replace")
        return (self.source / name).read_text(errors="replace")


def uid_from_name(name: str) -> str | None:
    match = UID_RE.search(name)
    return f"u{match.group(1)}" if match else None


def day_from_timestamp(value) -> str | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(ts) or ts <= 0:
        return None
    return datetime.fromtimestamp(ts).date().isoformat()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scale_1_to_10(value: float, old_min: float, old_max: float) -> float:
    if old_max == old_min:
        return 5.0
    return clamp(1 + ((value - old_min) / (old_max - old_min)) * 9, 1, 10)


def parse_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_json_records(text: str):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def record_day(record: dict) -> str | None:
    for key in ("resp_time", "timestamp", "time", "start", "start_timestamp"):
        if key in record:
            day = day_from_timestamp(record[key])
            if day:
                return day
    return None


def average(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def aggregate_sleep(reader: DatasetReader):
    daily = defaultdict(list)
    for name in reader.names():
        if "/EMA/response/Sleep/" not in name or not name.endswith(".json"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        for record in parse_json_records(reader.read_text(name)):
            day = record_day(record)
            hour = parse_float(record.get("hour"))
            if day and hour is not None:
                daily[(uid, day)].append(clamp(hour, 0, 14))
    return {key: average(values) for key, values in daily.items()}


def aggregate_stress(reader: DatasetReader):
    daily = defaultdict(list)
    for name in reader.names():
        if "/EMA/response/Stress/" not in name or not name.endswith(".json"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        for record in parse_json_records(reader.read_text(name)):
            day = record_day(record)
            level = parse_float(record.get("level"))
            if day and level is not None:
                daily[(uid, day)].append(scale_1_to_10(level, 1, 5))
    return {key: average(values) for key, values in daily.items()}


def aggregate_mood(reader: DatasetReader):
    daily = defaultdict(list)
    for name in reader.names():
        if "/EMA/response/Mood" not in name or not name.endswith(".json"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        for record in parse_json_records(reader.read_text(name)):
            day = record_day(record)
            happy = parse_float(record.get("happy"))
            sad = parse_float(record.get("sad"))
            values = []
            if happy is not None:
                values.append(scale_1_to_10(happy, 1, 3))
            if sad is not None:
                values.append(11 - scale_1_to_10(sad, 1, 3))
            if day and values:
                daily[(uid, day)].append(average(values))
    return {key: average(values) for key, values in daily.items()}


def aggregate_class_hours(reader: DatasetReader):
    daily = defaultdict(list)
    for name in reader.names():
        if "/EMA/response/Class/" not in name or not name.endswith(".json"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        for record in parse_json_records(reader.read_text(name)):
            day = record_day(record)
            hours = parse_float(record.get("hours"))
            if day and hours is not None:
                daily[(uid, day)].append(clamp(hours, 0, 14))
    return {key: sum(values) for key, values in daily.items()}


def aggregate_deadlines(reader: DatasetReader):
    deadline_load = {}
    names = [name for name in reader.names() if name.endswith("education/deadlines.csv")]
    if not names:
        return deadline_load

    rows = csv.DictReader(reader.read_text(names[0]).splitlines())
    for row in rows:
        uid = row.get("uid")
        if not uid:
            continue
        uid = uid.strip()
        if uid.isdigit():
            uid = f"u{uid.zfill(2)}"
        for day, raw in row.items():
            if day == "uid":
                continue
            value = parse_float(raw)
            if value is not None:
                deadline_load[(uid, day)] = value
    return deadline_load


def aggregate_activity(reader: DatasetReader):
    # Activity files can contain very high-frequency samples. Sampling every
    # 120th record preserves a daily active-ratio proxy while keeping the
    # preprocessing step fast enough for a lightweight research prototype.
    sample_stride = 120
    daily_counts = defaultdict(lambda: {"active": 0, "total": 0})
    for name in reader.names():
        if "/sensing/activity/" not in name or not name.endswith(".csv"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        try:
            rows = csv.DictReader(reader.read_text(name).splitlines())
        except Exception:
            continue
        for index, row in enumerate(rows):
            if index % sample_stride != 0:
                continue
            day = day_from_timestamp(row.get("timestamp") or row.get("time"))
            raw = row.get(" activity inference") or row.get("activity inference") or row.get("activity") or row.get(" inference")
            inference = parse_float(raw)
            if day and inference is not None:
                daily_counts[(uid, day)]["total"] += 1
                if int(inference) in {1, 2}:
                    daily_counts[(uid, day)]["active"] += 1

    minutes = {}
    for key, counts in daily_counts.items():
        if counts["total"]:
            active_ratio = counts["active"] / counts["total"]
            minutes[key] = round(clamp(active_ratio * 180, 0, 180), 1)
    return minutes


def aggregate_phone_lock(reader: DatasetReader):
    daily_events = defaultdict(int)
    student_counts = defaultdict(list)
    for name in reader.names():
        if "/sensing/phonelock/" not in name or not name.endswith(".csv"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        try:
            rows = csv.DictReader(reader.read_text(name).splitlines())
        except Exception:
            continue
        for row in rows:
            day = day_from_timestamp(row.get("start") or row.get("timestamp") or row.get("time"))
            if day:
                daily_events[(uid, day)] += 1

    for (uid, _), count in daily_events.items():
        student_counts[uid].append(count)

    screen_time = {}
    for key, count in daily_events.items():
        uid, _ = key
        max_count = max(student_counts[uid]) if student_counts[uid] else 1
        screen_time[key] = round(scale_1_to_10(count, 0, max_count), 1)
    return screen_time


def aggregate_conversation(reader: DatasetReader):
    daily_minutes = defaultdict(float)
    student_values = defaultdict(list)
    for name in reader.names():
        if "/sensing/conversation/" not in name or not name.endswith(".csv"):
            continue
        uid = uid_from_name(name)
        if not uid:
            continue
        try:
            rows = csv.DictReader(reader.read_text(name).splitlines())
        except Exception:
            continue
        for row in rows:
            start = parse_float(row.get("start_timestamp") or row.get("start"))
            end = parse_float(row.get(" end_timestamp") or row.get("end_timestamp") or row.get("end"))
            day = day_from_timestamp(start)
            if day and start is not None and end is not None and end > start:
                daily_minutes[(uid, day)] += (end - start) / 60

    for (uid, _), minutes in daily_minutes.items():
        student_values[uid].append(minutes)

    social = {}
    for key, minutes in daily_minutes.items():
        uid, _ = key
        max_minutes = max(student_values[uid]) if student_values[uid] else 1
        social[key] = round(scale_1_to_10(minutes, 0, max_minutes), 1)
    return social


def student_mean(data: dict, uid: str, default: float) -> float:
    values = [value for (student, _), value in data.items() if student == uid]
    return average(values) if values else default


def build_rows(reader: DatasetReader):
    sleep = aggregate_sleep(reader)
    stress = aggregate_stress(reader)
    mood = aggregate_mood(reader)
    class_hours = aggregate_class_hours(reader)
    deadlines = aggregate_deadlines(reader)
    activity = aggregate_activity(reader)
    phone_use = aggregate_phone_lock(reader)
    social = aggregate_conversation(reader)

    keys = set(sleep) | set(stress) | set(mood) | set(class_hours) | set(deadlines) | set(activity) | set(phone_use) | set(social)
    if not keys:
        raise SystemExit("No usable StudentLife records found. Check that the path points to dataset.zip or the extracted dataset folder.")

    rows = []
    for uid, day in sorted(keys):
        sleep_hours = sleep.get((uid, day), student_mean(sleep, uid, 7.0))
        stress_score = stress.get((uid, day), student_mean(stress, uid, 5.0))
        mood_score = mood.get((uid, day), student_mean(mood, uid, 5.5))
        activity_minutes = activity.get((uid, day), student_mean(activity, uid, 30.0))
        screen_time = phone_use.get((uid, day), student_mean(phone_use, uid, 5.5))
        social_score = social.get((uid, day), student_mean(social, uid, 5.5))
        academic_hours = class_hours.get((uid, day), 0.0)
        deadline_pressure = deadlines.get((uid, day), 0.0)
        work_hours = clamp(4.5 + academic_hours + deadline_pressure * 1.5, 0, 14)

        rows.append(
            {
                "student_id": uid,
                "date": day,
                "sleep": round(clamp(sleep_hours, 0, 14), 1),
                "stress": round(clamp(stress_score, 1, 10), 1),
                "mood": round(clamp(mood_score, 1, 10), 1),
                "energy": round(clamp((mood_score + (11 - stress_score)) / 2, 1, 10), 1),
                "screenTime": round(clamp(screen_time, 0, 24), 1),
                "workHours": round(work_hours, 1),
                "activity": round(clamp(activity_minutes, 0, 300), 1),
                "social": round(clamp(social_score, 1, 10), 1),
            }
        )
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 scripts/prepare_studentlife.py /path/to/dataset.zip [output.csv]")

    source = Path(sys.argv[1]).expanduser().resolve()
    output = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else Path("data/studentlife_wellbeing.csv").resolve()

    if not source.exists():
        raise SystemExit(f"Dataset not found: {source}")

    reader = DatasetReader(source)
    rows = build_rows(reader)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=APP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    students = sorted({row["student_id"] for row in rows})
    print(f"Wrote {len(rows)} rows for {len(students)} students to {output}")


if __name__ == "__main__":
    main()
