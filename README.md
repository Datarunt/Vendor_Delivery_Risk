# Vendor Delivery Risk Predictor

A machine learning system that predicts whether a vendor will deliver the quantity they committed to and flags them as HIGH, MED, or LOW risk before the delivery date arrives.

---

## What Problem Does This Solve?

In supply chain and procurement, vendors make delivery commitments (a date and a quantity). The question this system answers is:

> **"Based on this vendor's history, how likely are they to actually deliver what they promised?"**

Instead of finding out a vendor is unreliable after a missed delivery, this tool gives the team an early warning so they can follow up, find alternatives, or adjust plans proactively.

---

## How It Works — Plain English

```
Raw delivery history (Excel/CSV)
        ↓
  data_processing.py       ← cleans and prepares the data
        ↓
  model.py                 ← XGBoost predicts expected delivery quantity
        ↓
  risk_engine.py           ← converts forecast into a risk score (HIGH / MED / LOW)
        ↓
  forecast_build_services.py  ← runs the forecast for each vendor/material
  forecast_accuracy_services.py ← measures how accurate past forecasts were
        ↓
  combine_services.py      ← assembles everything into one output
        ↓
  upload_services.py       ← saves results to Excel
        ↓
  app.py                   ← the web interface that ties it all together
        ↓
  performance_check.py     ← after deliveries happen, scores how well the model did
```

---

## File-by-File Breakdown

### `app.py` — The Front Door
This is the file you run. It launches a web app in your browser (built with Flask) where you upload your data files and get a risk report back. Everything else is called from here behind the scenes.

---

### `data_processing.py` — Data Cleaning
**What it does:** Takes your raw historical delivery file (Excel or CSV) and cleans it up so the model can use it.

Specifically it:
- Keeps only the columns that matter (vendor name, material, dates, quantities, lateness)
- Fixes date and number formatting
- Handles a nuance: if a vendor delivered MORE than required, their "days late" gets set to 0 (over-delivery is not a failure)
- Calculates each vendor's average days late across all their deliveries
- Figures out which vendor is currently assigned to each material (in case it changed over time)

**Input:** Raw Excel or CSV file from your ERP/purchasing system

**Output:** A clean dataframe ready for the model

---

### `model.py` — The Prediction Engine
**What it does:** Uses XGBoost (a machine learning algorithm) to predict how much quantity a vendor is likely to deliver on a future date.

It learns from four things:
1. **Trend** — are deliveries getting better or worse over time?
2. **Delivery description score** — how bad were past late deliveries? (On time = 0, >15 days late = 20)
3. **Vendor identity** — which vendor is this?
4. **Seasonality** — does delivery performance change by month?

**Output:** Two numbers per vendor/material:
- `mu` — the predicted delivery quantity
- `sigma` — the uncertainty (how much the prediction could be off)

Think of these as: *"We expect 85 units, give or take 12."*

---

### `risk_engine.py` — The Risk Scorer
**What it does:** Takes the prediction from `model.py` and the vendor's committed quantity, then calculates the probability they'll actually deliver enough.

It uses a statistical method (normal distribution) to answer: *"Given our forecast and its uncertainty, what are the odds the actual delivery meets or exceeds the commitment?"*

**Risk labels:**
| Probability of meeting commitment | Risk Label |
|---|---|
| Less than 60% | 🔴 HIGH |
| 60% – 85% | 🟡 MED |
| Greater than 85% | 🟢 LOW |

It also produces a **confidence interval** — a range the actual delivery is likely to fall within (e.g. "between 72 and 98 units").

---

### `forecast_build_services.py` — Forecast Runner
**What it does:** Loops through every material/vendor combination and calls `model.py` for each one. Handles edge cases like vendors with too few data points to forecast (minimum 5 historical rows required).

---

### `forecast_accuracy_services.py` — Forecast Accuracy Tracker
**What it does:** For forecasts that have already come true (past dates), it compares what the model predicted vs what actually arrived. Calculates R² to show how well the quantity forecast matched reality.

---

### `combine_services.py` — The Assembler
**What it does:** Pulls together the forecast output, risk scores, vendor metadata, and accuracy metrics into a single unified dataframe that becomes the final report.

---

### `upload_services.py` — File Saver
**What it does:** Takes the final combined output and writes it to an Excel file with a formatted "Risk Output" sheet. This is the file the team uses to review vendor risk.

---

### `performance_check.py` — Model Scorecard
**What it does:** After actual deliveries have happened, this script compares the model's risk predictions against what really occurred. Run this separately to evaluate model accuracy.

It works by:
1. You pick your prediction file and your historical receipts file via a popup
2. It calculates each vendor's fulfillment ratio (actual received ÷ committed)
3. It derives what the risk label *should* have been based on that ratio
4. It compares predicted vs actual risk labels

**Output:** An Excel file with four sheets:
- **Risk Output** — full results with actuals appended
- **Confusion Matrix** — shows where the model got it right/wrong by risk tier
- **Metrics** — overall accuracy, HIGH risk precision, recall, and F1 score
- **Classification Report** — detailed breakdown by risk category

---

## Current Model Performance

Based on 428 evaluated records:

| Metric | Value |
|---|---|
| Overall Accuracy | 33% |
| HIGH Risk Precision | 38% |
| HIGH Risk Recall | 66% |
| HIGH Risk F1 | 48% |

**What this means in plain terms:** The model catches about 2 out of 3 vendors that are genuinely high risk (good recall). However, it also flags many low-risk vendors as high risk (lower precision). The recommended use is as a **prioritization tool** focus attention on HIGH-flagged vendors, but don't treat the label as a final verdict without human review.

---

## Setup

**Install dependencies:**
```bash
pip install pandas numpy xgboost scipy scikit-learn flask xlsxwriter openpyxl
```

**Run the app:**
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

---

## Input Data Requirements

Your historical delivery file needs these columns:

| Column | Description |
|---|---|
| Name 1 | Vendor name |
| Material | Material/part number |
| Date Due | When delivery was due |
| Date Rcvd | When delivery actually arrived |
| Qty Due | Quantity ordered |
| Qty Rcvd | Quantity actually received |
| Description | Delivery outcome (e.g. "Hit / On Time", "Miss / 1 Day Late") |
| Days Late | Number of days late (0 if on time) |
| Net Order Price | Order value |
| Description p. group | Purchasing group category |

---

## Known Limitations

- Needs at least 5 historical delivery records per vendor/material to generate a forecast
- MED risk category is underrepresented in training data, making MED predictions less reliable than HIGH/LOW
- Vendor encoding does not yet carry historical performance signal — a planned improvement is to encode vendors by their actual late delivery rate rather than an arbitrary number
- Model is retrained on every run; no persistent model file is saved between sessions

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Core forecasting model |
| SciPy | Probability calculations for risk scoring |
| Pandas / NumPy | Data processing |
| Flask | Web interface |
| Scikit-learn | Model evaluation metrics |
| XlsxWriter | Excel output |
