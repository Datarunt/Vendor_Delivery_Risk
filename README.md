# Vendor Delivery Risk Predictor

A Python/Flask tool that compares each vendor commit (a date and a quantity) against an XGBoost forecast of the quantity that material is likely to arrive in, and labels the commit **HIGH** or **LOW** risk before the commit date arrives.

> **Status: under active development.** The forecast is a *quantity* model. It does not predict whether a delivery arrives on time, and in the saved evaluation runs the HIGH flag was not more accurate than the overall share of HIGH outcomes (see [Model Performance](#model-performance)). Treat the output as experimental, not as a verdict on a vendor.

---

## What Problem Does This Solve?

Vendors make delivery commitments (a date and a quantity). The question this tool tries to answer is:

> **"Given the recent delivery quantities for this material and vendor, how likely is it that the quantity arriving around the commit date meets the quantity committed?"**

The goal is an early warning so the team can follow up, find alternatives, or adjust plans before a shortfall happens.

---

## How It Works

In plain terms:

1. Historical delivery data is cleaned, and each past delivery gets a lateness score.
2. For every Material + Vendor pair with at least 5 past deliveries, an XGBoost model is trained on past delivered **quantities** and forecasts a quantity for each day of the forecast window (`mu`). It also produces one uncertainty value (`sigma`): the spread of the model's errors on its own training data.
3. Each vendor commit inside the window is matched to the forecast for the same material and date (rolled up to the week or month if selected).
4. The tool computes the probability that a normal distribution centered on `mu` with spread `sigma` reaches the committed quantity. Below 75% is flagged HIGH risk.

```
app.py  (Flask web UI: upload files, pick start date / window / aggregation)
   |
   |  "Predict Risk" button
   v
upload_services.py            <- reads the uploads, appends optional OTD data,
   |                             runs the forecast, loads commits + owner matrix
   |-- data_processing.py     <- cleans the historical file
   |-- forecast_build_services.py   <- loops over Material + Vendor pairs
   |      |-- model.py                     <- XGBoost quantity forecast + sigma
   |      '-- forecast_accuracy_services.py <- holdout backtest (accuracy metrics)
   v
forecast (in memory, downloadable as CSV) + accuracy metrics shown in the UI
   |
   |  "Combine" button
   v
combine_services.py           <- matches commits to forecasts, adds buyer/vendor info
   '-- risk_engine.py         <- probability, HIGH/LOW risk, interval, confidence
   |
   v
app.py writes vendor_commit_risk.xlsx  ("Risk Output" + "Accuracy" sheets)

performance_check.py          <- run separately, later, to score past predictions
vendor-commit-eda.py          <- optional, standalone exploratory analysis
```

---

## Using the App

1. Run `python app.py`. A browser tab opens at `http://127.0.0.1:5000`.
2. Upload your files (see [Input Files](#input-files)). Historical File Type 1 and Type 2 are each optional, but at least one is required. The Portal Commit file and Owner Matrix are required.
3. For each uploaded file, map your column names to the required fields. Columns whose names already match are pre-selected.
4. Set the **Forecast Start Date**, the **Forecast Window** (1 to 90 days, default 14), and the **Commit Aggregation** (day, week, or month).
5. Click **Predict Risk**. The forecast is built and the forecast accuracy metrics appear (AVG Commit Qty, R², MAE, RMSE, MAPE). You can download the raw forecast as a CSV.
6. Click **Combine Forecast, Commits & Owner Matrix** to download `vendor_commit_risk.xlsx`.

Results are held in memory for the running server process, so the app is built for one user at a time.

---

## Input Files

### Historical Performance File, Type 1 (optional)

| Required field | Description |
|---|---|
| Vendor | Vendor name (renamed to `Vendor Name` internally) |
| Material | Material / part number |
| Date Received | When the delivery arrived |
| Quantity Due | Quantity ordered |
| Quantity Received | Quantity actually received |
| Number of Days Late | Days late (negative = early) |

The UI also lists `Days Late Classification`, but the loader always derives the classification from `Number of Days Late`, so a mapped column is ignored.

### Historical Performance File, Type 2 / "New OTD" file (optional)

| Required field | Description |
|---|---|
| Material | Material / part number |
| VENDOR_NAME, VENDOR_ID | Vendor name and ID (leading zeros stripped from the ID) |
| Stat Date | Start date used for the days-late calculation |
| Delivered in Full Y/N | Whether the line was delivered in full |
| Delivered in Full Date | Date it was delivered in full |
| Month of OTD Measure | Month the miss was measured in |
| DateDiff Measured Month | Number of months measured (for misses) |
| Received Qty | Quantity received |

These rows are converted into the Type 1 format and appended to the historical data:

- **Days late:** if delivered in full, `Delivered in Full Date - Stat Date`. Otherwise, the days from `Stat Date` through the end of the last measured month.
- **Quantity:** `Received Qty` is used for both Quantity Due and Quantity Received, so over- and under-delivery are not captured for these rows.

### Portal Commit File (required)

| Field | Description |
|---|---|
| Material | Material / part number |
| Vendor | Vendor ID |
| Commit Dt by Suppl | Date the vendor committed to deliver |
| Commit Qty | Quantity the vendor committed |
| Vendor Name | Optional. If missing, the name comes from the Owner Matrix, then falls back to the vendor ID |

### Owner Matrix File (required)

| Field | Description |
|---|---|
| Supplier # | Vendor ID (matched to `Vendor` in the commit file after stripping leading zeros) |
| Assigned Buyer | Buyer responsible for the vendor |
| Supplier Name or Vendor Name | Optional. Used to fill in vendor names |

---

## File-by-File Breakdown

### `app.py` - The Front Door
Flask web app. Routes: `/` (upload form), `/get_columns` (reads a file's headers for the column-mapping table), `/run_forecast` (remaps columns, runs `upload_services`, stores the results), `/download_forecast.csv`, and `/combine` (runs `combine_services` and writes the Excel file with the **Risk Output** and **Accuracy** sheets). The server starts on port 5000 with `debug=True` and `host="0.0.0.0"`.

### `upload_services.py` - Ingestion and Forecast Orchestration
Loads the historical file (if provided), converts and appends the OTD file (if provided), calls `build_forecast`, then reads the commit and owner matrix files. Returns the forecast, commits, and owner matrix as CSV bytes plus the accuracy metrics. It does **not** write the final Excel file (that happens in `app.py`).

### `data_processing.py` - Data Cleaning
Cleans the Type 1 historical file:
- Keeps only the needed columns and renames `Vendor` to `Vendor Name`
- Normalizes material numbers (takes the first token and strips trailing letters)
- Fixes date and number formats
- Derives `Days Late Classification` from `Number of Days Late`: Early, On Time, 1 Day Late, 2-4 Days, 5-15 Days, >15 Days
- Sets `Number of Days Late` to 0 when `Quantity Received >= Quantity Due` (over-delivery is not a failure). This happens *after* the classification is derived, so an over-delivered but late row keeps its late classification
- Drops rows with no days-late value or no received date
- Adds `Avg Days Late`: each vendor's mean days late across all of their rows

### `model.py` - The Forecast Model
Trains an XGBoost regressor (50 trees, depth 3, learning rate 0.1, seed 42) on one Material + Vendor pair's delivery history. The target is `Quantity Received`. Features:

| Feature | In training | At forecast time |
|---|---|---|
| Trend | Days since the first delivery | Days since the first delivery (extends past the training range) |
| Delivery description score | Lateness score of *that same delivery*: on time 0, 1 day late 1, 2-4 days 3, 5-15 days 10, >15 days 20 (early = 0) | Set to the historical average |
| Vendor performance | Vendor's `Avg Days Late` | Set to the historical average |
| Seasonality | Month of the delivery | Month of the forecast date |

**Output:** an array of forecast quantities (`mu`, floored at 0), one per day of the window, plus a single `sigma` (standard deviation of the training residuals). Because the description score and vendor value are constants at forecast time, the forecast varies across the window only through the date trend and month.

### `forecast_build_services.py` - Forecast Runner
Builds the list of future dates, runs the accuracy backtest, and loops through every Material + Vendor pair calling `model.py`. Pairs with fewer than 5 historical rows (or any other error) are skipped silently. Output rows are labeled with the *latest* vendor seen for the material.

### `forecast_accuracy_services.py` - Forecast Accuracy Tracker
A holdout backtest: the last `horizon` days of history are held out, the model is trained on everything earlier (per material), and its forecasts are compared with the deliveries that actually arrived on matching material/date rows. Reports:

- `AVG_COMMIT_QTY` (mean actual quantity of the matched rows), `MAE`, `RMSE`, `MAPE`, `R2`

MAE, RMSE, and MAPE only count under-prediction (shortfalls); over-prediction counts as zero error. The result is one set of numbers for the whole run, not per vendor. If nothing matches, all values are "N/A".

### `combine_services.py` - The Assembler
- Filters commits and forecasts to the forecast window and rolls dates up by the chosen aggregation: **day** (no change), **week** (coming Sunday), **month** (month end)
- Commit quantities are summed per Material + bucket
- For week and month buckets, the forecast is taken from the **single latest forecast day in the bucket** (daily forecasts are not summed)
- Matches commits to forecasts on Material + date; if several forecasts share a material and date, the one with the lowest sigma is kept
- Fills in vendor name and assigned buyer from the commit file and Owner Matrix
- Calls `risk_engine.py` and returns the final table
- **Commits with no matching forecast are dropped** from the output (a warning is printed to the console)

### `risk_engine.py` - The Risk Scorer
For each commit: `probability = P(Normal(mu, sigma) >= Commit Qty)`, rounded to a whole percent.

| Probability of meeting commitment | Risk |
|---|---|
| Below 75% | 🔴 HIGH |
| 75% or above | 🟢 LOW |

The interval is `mu ± 1.96 × sigma`, floored at 0. A separate **Confidence** level is based on the interval's relative width, `(upper - lower) / mu`:

| Relative width | Confidence |
|---|---|
| 0.0356 or less | HIGH |
| Up to 0.1227 | MED |
| Above 0.1227 | LOW |

These two cutoffs are hard-coded constants derived from the tertiles of one dataset. They should be re-derived when the data changes.

Special cases: if `mu`, `sigma`, or `Commit Qty` is missing, the row is scored 0% / LOW. If `sigma` is 0, probability is set to 50%, which is HIGH.

### `performance_check.py` - Model Scorecard
Run separately (`python performance_check.py`) after deliveries have happened. File dialogs ask for:

1. The prediction file (`vendor_commit_risk.xlsx`, read from its **Risk Output** sheet)
2. A receipts file with `Material`, `Document Date`, and `Quantity` columns (a different format from the historical input file)

For each commit row, receipts are counted up to the commit date, with each period starting after the previous commit date for that material. Surplus and shortfall carry forward to the next period. Then:

- `Fulfillment_Ratio = actual received / committed`
- `Actual_Risk = LOW if the ratio is 1.0 or more, otherwise HIGH`
- The predicted `Risk` is compared with `Actual_Risk`

Only commits dated inside the receipts file's date range are scored. **Output:** an Excel file with **Risk Output** (with actuals appended), **Confusion Matrix** (HIGH/LOW), **Metrics** (records, accuracy, HIGH precision, recall, F1), and **Classification Report**.

### `vendor-commit-eda.py` - Exploratory Analysis (optional)
A standalone script. It opens a file picker and a column-selection window, then writes a Word (`.docx`) report of tables and control charts. It is not used by the app.

---

## Output

`vendor_commit_risk.xlsx`:

- **Risk Output:** Material, Vendor Name, Vendor Commit, Commit Qty, Probability, Risk, Confidence Interval, Confidence, Assigned Buyer
- **Accuracy:** the backtest metrics from `forecast_accuracy_services.py`

---

## Model Performance

Results from saved evaluation runs in `Data/Performance/`. Runs use different data and code versions, so re-run `performance_check.py` after any change.

| Run file | Records | Classes | Accuracy | HIGH precision | HIGH recall | Share of rows actually HIGH |
|---|---|---|---|---|---|---|
| `4.27.risk_output_with_actuals.xlsx` | 428 | 3 (HIGH/MED/LOW) | 33.2% | 37.6% | 66.3% | 39.5% |
| `risk_output_with_actuals.99.xlsx` | 456 | 2 | 41.5% | 39.6% | 65.3% | 43.6% |
| `risk_output_with_actuals.z.xlsx` | 349 | 2 | 47.0% | 20.9% | 9.6% | 41.8% |

**How to read this:** precision is the share of HIGH-flagged commits that turned out to be HIGH. In every run above it is *below* the overall share of HIGH outcomes, so a HIGH flag does not currently identify riskier commits than an average commit. The first run's 3-class accuracy is not comparable with the later 2-class runs (MED was removed from the risk labels). Until this improves, the labels should not be used to prioritize vendors.

---

## Known Limitations

**Modeling**
- The model forecasts delivery **quantity**. Nothing in it predicts whether a delivery arrives by the commit date, but `performance_check.py` scores quantity received by the commit date, so timing affects the scoring and not the prediction.
- The vendor performance feature is a per-vendor average, and each model is trained on a single Material + Vendor pair, so it is the same value on every training row and carries no signal. OTD rows do not get an `Avg Days Late` value at all.
- The delivery description score is the lateness of the same delivery whose quantity is being predicted, not earlier history, and it is replaced by the average at forecast time.
- `sigma` is measured on the training data, so it underestimates true forecast error, and it is the same for every day of the window. A walk-forward (out-of-sample) sigma is planned.
- Tree models cannot extrapolate the date trend beyond the training range.
- A vendor needs at least 5 historical rows per Material + Vendor pair; otherwise it is skipped with no message.
- The model is retrained on every run (and again for the accuracy backtest). No model file is saved.

**Risk engine**
- Missing inputs are scored 0% / LOW, which reads as a confident "low risk" rather than "unknown".
- `sigma = 0` forces 50% probability and therefore HIGH.
- The confidence cutoffs are fixed constants (see above).

**Data handling**
- For materials with more than one vendor, forecasts are matched on Material + date only (lowest sigma wins), and forecast rows are labeled with the latest vendor. This can attach one vendor's forecast to another vendor's commit.
- Material numbers are normalized differently in the historical file (trailing letters stripped) and the commit file (trimmed and upper-cased only). Materials that don't match after these steps get no forecast.
- Commits with no matching forecast are dropped from the output.
- The accuracy metrics ignore over-prediction and are computed once per run.

**Operations**
- Session state lives in module-level variables, so only one user can use a running server at a time.
- The dev server binds to all network interfaces with `debug=True`. Run it only on a trusted network.

---

## Setup

Developed on Python 3.11.

```bash
pip install pandas numpy xgboost scipy scikit-learn flask xlsxwriter openpyxl
```

For `vendor-commit-eda.py`, also install `matplotlib` and `python-docx`. `performance_check.py` and the EDA script use `tkinter` file dialogs (included with most Python installs; on Linux it may need the `python3-tk` package).

```bash
python app.py
```

Then open `http://localhost:5000` if the browser doesn't open on its own.

---

## Repository Layout

| Path | Contents |
|---|---|
| `app.py`, `*_services.py`, `data_processing.py`, `model.py`, `risk_engine.py` | Application code |
| `performance_check.py` | Scores past predictions against actual receipts |
| `vendor-commit-eda.py` | Exploratory analysis script |
| `templates/index.html` | Web UI |
| `Data/` | Working files: Historical, Commits, Account Matrix, Actuals, Predictions, Performance, EDA |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Quantity forecasting model |
| SciPy | Normal-distribution probability calculation |
| Pandas / NumPy | Data processing |
| Flask | Web interface |
| Scikit-learn | Evaluation metrics (`performance_check.py`) |
| XlsxWriter / openpyxl | Excel output and input |
| matplotlib / python-docx | EDA report (optional) |
