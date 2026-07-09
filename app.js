const STORAGE_KEY = "personal-wellbeing-time-series-entries";

const fields = {
  date: document.querySelector("#date"),
  sleep: document.querySelector("#sleep"),
  stress: document.querySelector("#stress"),
  mood: document.querySelector("#mood"),
  energy: document.querySelector("#energy"),
  screenTime: document.querySelector("#screenTime"),
  workHours: document.querySelector("#workHours"),
  activity: document.querySelector("#activity"),
  social: document.querySelector("#social"),
};

const outputs = {
  todayScore: document.querySelector("#todayScore"),
  riskBadge: document.querySelector("#riskBadge"),
  todayMessage: document.querySelector("#todayMessage"),
  avgScore: document.querySelector("#avgScore"),
  entryCount: document.querySelector("#entryCount"),
  stressTrend: document.querySelector("#stressTrend"),
  sleepTrend: document.querySelector("#sleepTrend"),
  studentSelect: document.querySelector("#studentSelect"),
  physicalScore: document.querySelector("#physicalScore"),
  mentalScore: document.querySelector("#mentalScore"),
  socialScore: document.querySelector("#socialScore"),
  occupationalScore: document.querySelector("#occupationalScore"),
  digitalScore: document.querySelector("#digitalScore"),
  chart: document.querySelector("#chart"),
  explanations: document.querySelector("#explanations"),
};

const rangePairs = [
  ["stress", "stressValue"],
  ["mood", "moodValue"],
  ["energy", "energyValue"],
  ["social", "socialValue"],
];

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function loadEntries() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveEntries(entries) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function calculateScore(entry) {
  const domains = calculateDomainScores(entry);
  return Math.round(
    domains.physical * 0.24 +
      domains.mental * 0.28 +
      domains.social * 0.14 +
      domains.occupational * 0.17 +
      domains.digital * 0.17
  );
}

function calculateDomainScores(entry) {
  const sleepScore = clamp((entry.sleep / 8) * 100, 0, 100);
  const stressScore = 100 - (entry.stress - 1) * 11.1;
  const moodScore = entry.mood * 10;
  const energyScore = entry.energy * 10;
  const socialScore = entry.social * 10;
  const activityScore = clamp((entry.activity / 45) * 100, 0, 100);
  const screenScore = 100 - clamp(Math.max(entry.screenTime - 5, 0) * 10, 0, 60);
  const workScore = 100 - clamp(Math.max(entry.workHours - 8, 0) * 12, 0, 60);

  return {
    physical: Math.round(sleepScore * 0.45 + activityScore * 0.3 + energyScore * 0.25),
    mental: Math.round(stressScore * 0.5 + moodScore * 0.35 + energyScore * 0.15),
    social: Math.round(socialScore),
    occupational: Math.round(workScore * 0.55 + stressScore * 0.3 + energyScore * 0.15),
    digital: Math.round(screenScore * 0.7 + sleepScore * 0.15 + stressScore * 0.15),
  };
}

function riskLevel(score) {
  if (score < 50) return { label: "High risk", className: "high" };
  if (score < 70) return { label: "Watch", className: "warning" };
  return { label: "Stable", className: "" };
}

function explanationFor(entry, previous) {
  const items = [];
  const domains = calculateDomainScores(entry);
  const lowestDomain = Object.entries(domains).sort((a, b) => a[1] - b[1])[0];

  if (lowestDomain && lowestDomain[1] < 65) {
    items.push(`The lowest domain today is ${lowestDomain[0]} wellbeing (${lowestDomain[1]}/100).`);
  }

  if (entry.sleep < 6) {
    items.push("Sleep is below 6 hours, which strongly lowers the physical wellbeing domain.");
  }
  if (entry.stress >= 8) {
    items.push("Stress is high, suggesting pressure in the mental or occupational domain.");
  }
  if (entry.screenTime > 8) {
    items.push("Screen time is above 8 hours, increasing digital-load risk.");
  }
  if (entry.activity < 20) {
    items.push("Physical activity is low, which may reduce energy and recovery.");
  }
  if (entry.social <= 4) {
    items.push("Social connection is low, suggesting reduced social wellbeing.");
  }
  if (entry.workHours > 9) {
    items.push("Study or work hours are high, which may increase burnout risk.");
  }

  if (previous) {
    const scoreDelta = calculateScore(entry) - calculateScore(previous);
    if (scoreDelta <= -8) {
      items.unshift(`Wellbeing dropped by ${Math.abs(scoreDelta)} points compared with the previous entry.`);
    } else if (scoreDelta >= 8) {
      items.unshift(`Wellbeing improved by ${scoreDelta} points compared with the previous entry.`);
    }
  }

  if (!items.length) {
    items.push("Signals look balanced today. Sleep, stress, mood, activity, and social connection are not showing major risk patterns.");
  }

  return items;
}

function trend(entries, field) {
  const recent = entries.slice(-7);
  if (recent.length < 2) return "--";
  const first = recent[0][field];
  const last = recent[recent.length - 1][field];
  const delta = last - first;
  if (Math.abs(delta) < 0.5) return "Stable";
  return delta > 0 ? `Up ${delta.toFixed(1)}` : `Down ${Math.abs(delta).toFixed(1)}`;
}

function renderChart(entries) {
  outputs.chart.innerHTML = "";

  if (!entries.length) {
    outputs.chart.innerHTML = `<div class="empty-state">No entries yet</div>`;
    return;
  }

  const visible = entries.slice(-21);
  const width = outputs.chart.clientWidth || 800;
  const height = outputs.chart.clientHeight || 280;
  const gap = width / (visible.length + 1);

  visible.forEach((entry, index) => {
    const score = calculateScore(entry);
    const barHeight = Math.max(6, (score / 100) * (height - 80));
    const level = riskLevel(score);
    const x = gap * (index + 1);

    const bar = document.createElement("div");
    bar.className = `bar ${level.className}`;
    bar.style.left = `${x - 9}px`;
    bar.style.height = `${barHeight}px`;
    bar.title = `${entry.date}: ${score}`;

    const label = document.createElement("span");
    label.className = "bar-label";
    label.style.left = `${x}px`;
    label.textContent = entry.date.slice(5);

    outputs.chart.append(bar, label);
  });
}

function updateStudentOptions(entries) {
  const current = outputs.studentSelect.value || "all";
  const students = Array.from(new Set(entries.map((entry) => entry.student_id || "manual"))).sort();

  outputs.studentSelect.innerHTML = `<option value="all">All students</option>`;
  students.forEach((student) => {
    const option = document.createElement("option");
    option.value = student;
    option.textContent = student === "manual" ? "Manual entries" : student;
    outputs.studentSelect.append(option);
  });

  outputs.studentSelect.value = current === "all" || students.includes(current) ? current : "all";
}

function render() {
  const entries = loadEntries().sort((a, b) => a.date.localeCompare(b.date));
  updateStudentOptions(entries);
  const selectedStudent = outputs.studentSelect.value;
  const visibleEntries = selectedStudent === "all"
    ? entries
    : entries.filter((entry) => (entry.student_id || "manual") === selectedStudent);
  const latest = visibleEntries[visibleEntries.length - 1];
  const previous = visibleEntries[visibleEntries.length - 2];

  outputs.entryCount.textContent = `${visibleEntries.length} ${visibleEntries.length === 1 ? "entry" : "entries"}`;

  if (!latest) {
    outputs.todayScore.textContent = "--";
    outputs.riskBadge.textContent = "No data";
    outputs.riskBadge.className = "risk-badge";
    outputs.todayMessage.textContent = "Add an entry or load sample data to begin.";
    outputs.avgScore.textContent = "--";
    outputs.stressTrend.textContent = "--";
    outputs.sleepTrend.textContent = "--";
    outputs.physicalScore.textContent = "--";
    outputs.mentalScore.textContent = "--";
    outputs.socialScore.textContent = "--";
    outputs.occupationalScore.textContent = "--";
    outputs.digitalScore.textContent = "--";
    outputs.explanations.innerHTML = "<li>Add data to generate an explanation.</li>";
    renderChart([]);
    return;
  }

  const latestScore = calculateScore(latest);
  const domains = calculateDomainScores(latest);
  const level = riskLevel(latestScore);
  const avg = Math.round(visibleEntries.reduce((sum, entry) => sum + calculateScore(entry), 0) / visibleEntries.length);

  outputs.todayScore.textContent = latestScore;
  outputs.riskBadge.textContent = level.label;
  outputs.riskBadge.className = `risk-badge ${level.className}`;
  outputs.todayMessage.textContent = `Latest entry: ${latest.date}.`;
  outputs.avgScore.textContent = avg;
  outputs.stressTrend.textContent = trend(visibleEntries, "stress");
  outputs.sleepTrend.textContent = trend(visibleEntries, "sleep");
  outputs.physicalScore.textContent = domains.physical;
  outputs.mentalScore.textContent = domains.mental;
  outputs.socialScore.textContent = domains.social;
  outputs.occupationalScore.textContent = domains.occupational;
  outputs.digitalScore.textContent = domains.digital;
  outputs.explanations.innerHTML = explanationFor(latest, previous)
    .map((item) => `<li>${item}</li>`)
    .join("");

  renderChart(visibleEntries);
}

function exportCsv() {
  const entries = loadEntries().sort((a, b) => a.date.localeCompare(b.date));
  if (!entries.length) return;

  const headers = [
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
    "physicalScore",
    "mentalScore",
    "socialScore",
    "occupationalScore",
    "digitalScore",
    "wellbeingScore",
  ];

  const rows = entries.map((entry) => {
    const domains = calculateDomainScores(entry);
    const row = {
      ...entry,
      physicalScore: domains.physical,
      mentalScore: domains.mental,
      socialScore: domains.social,
      occupationalScore: domains.occupational,
      digitalScore: domains.digital,
      wellbeingScore: calculateScore(entry),
    };
    return headers.map((header) => row[header]).join(",");
  });

  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "personal-wellbeing-time-series.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((header) => header.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index]?.trim() ?? "";
    });
    return row;
  });
}

function normalizeImportedEntry(row) {
  return {
    student_id: row.student_id || "imported",
    date: row.date,
    sleep: Number(row.sleep),
    stress: Number(row.stress),
    mood: Number(row.mood),
    energy: Number(row.energy),
    screenTime: Number(row.screenTime),
    workHours: Number(row.workHours),
    activity: Number(row.activity),
    social: Number(row.social),
  };
}

function validEntry(entry) {
  return (
    entry.date &&
    Number.isFinite(entry.sleep) &&
    Number.isFinite(entry.stress) &&
    Number.isFinite(entry.mood) &&
    Number.isFinite(entry.energy) &&
    Number.isFinite(entry.screenTime) &&
    Number.isFinite(entry.workHours) &&
    Number.isFinite(entry.activity) &&
    Number.isFinite(entry.social)
  );
}

function formEntry() {
  return {
    student_id: "manual",
    date: fields.date.value,
    sleep: Number(fields.sleep.value),
    stress: Number(fields.stress.value),
    mood: Number(fields.mood.value),
    energy: Number(fields.energy.value),
    screenTime: Number(fields.screenTime.value),
    workHours: Number(fields.workHours.value),
    activity: Number(fields.activity.value),
    social: Number(fields.social.value),
  };
}

function generateSampleData() {
  const entries = [];
  const now = new Date();

  for (let i = 20; i >= 0; i -= 1) {
    const date = new Date(now);
    date.setDate(now.getDate() - i);
    const pressure = i < 7 ? 1.5 : 0;
    const recovery = i < 3 ? 1 : 0;
    entries.push({
      student_id: "demo",
      date: date.toISOString().slice(0, 10),
      sleep: clamp(7.5 - pressure + recovery * 0.7 + Math.sin(i) * 0.4, 4.5, 9),
      stress: Math.round(clamp(4 + pressure * 2 - recovery + Math.cos(i) * 1.2, 1, 10)),
      mood: Math.round(clamp(7 - pressure + recovery + Math.sin(i / 2), 1, 10)),
      energy: Math.round(clamp(7 - pressure + recovery + Math.cos(i / 3), 1, 10)),
      screenTime: clamp(5.5 + pressure * 2 - recovery + Math.sin(i / 2), 2, 11),
      workHours: clamp(6.5 + pressure * 2 - recovery * 0.5, 3, 12),
      activity: Math.round(clamp(45 - pressure * 15 + recovery * 20 + Math.sin(i) * 8, 5, 90)),
      social: Math.round(clamp(7 - pressure + recovery + Math.cos(i / 2), 1, 10)),
    });
  }

  return entries;
}

function updateRangeLabels() {
  rangePairs.forEach(([inputId, labelId]) => {
    document.querySelector(`#${labelId}`).textContent = fields[inputId].value;
  });
}

document.querySelector("#entryForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const entries = loadEntries();
  const entry = formEntry();
  const nextEntries = entries
    .filter((item) => item.date !== entry.date || (item.student_id || "manual") !== entry.student_id)
    .concat(entry);
  saveEntries(nextEntries);
  render();
});

document.querySelector("#loadSampleBtn").addEventListener("click", () => {
  saveEntries(generateSampleData());
  render();
});

outputs.studentSelect.addEventListener("change", render);

document.querySelector("#exportCsvBtn").addEventListener("click", exportCsv);

document.querySelector("#importCsvBtn").addEventListener("click", () => {
  document.querySelector("#importCsvInput").click();
});

document.querySelector("#importCsvInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const imported = parseCsv(text).map(normalizeImportedEntry).filter(validEntry);
  if (imported.length) {
    saveEntries(imported);
    render();
  }
  event.target.value = "";
});

document.querySelector("#clearDataBtn").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  render();
});

rangePairs.forEach(([inputId]) => {
  fields[inputId].addEventListener("input", updateRangeLabels);
});

fields.date.value = todayISO();
updateRangeLabels();
render();
