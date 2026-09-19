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

1. **Historical Commits** (past commits and what actually arrived against them) are cleaned. Each past delivery gets a lateness value, calculated from its commit date and actual delivery date, and a lateness score.
2. For every Material + Vendor pair with at least 5 past deliveries, an XGBoost model is trained on past delivered **quantities** and forecasts a quantity for each day of the forecast window (`mu`). It also produces one uncertainty value (`sigma`): the spread of the model's errors on its own training data.
3. Each **Current Commit** inside the window is matched to the forecast for the same material and date (rolled up to the week or month if selected).
4. The tool computes the probability that a normal distribution centered on `mu` with spread `sigma` reaches the committed quantity. Below 75% is flagged HIGH risk.
5. Optionally, each commit is matched to the **Owner Matrix** by vendor code and/or vendor name to add the Assigned Buyer.

```
app.py  (Flask web UI: upload files, map columns, pick start date / window / aggregation)
   |
   |  "Predict Risk" button
   v
upload_services.py            <- reads the uploads, runs the forecast,
   |                             loads Current Commits + Owner Matrix
   |-- data_processing.py     <- cleans Historical Commits, calculates days late
   |-- forecast_build_services.py   <- loops over Material + Vendor pairs
   |      |-- model.py                     <- XGBoost quantity forecast + sigma
   |      '-- forecast_accuracy_services.py <- holdout backtest (accuracy metrics)
   v
forecast (in memory, downloadable as CSV) + accuracy metrics shown in the UI
   |
   |  "Combine" button
   v
combine_services.py           <- matches commits to forecasts and to the Owner Matrix
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
2. Upload the three files. All are required:
   - **Historical Commits**
   - **Current Commits**
   - **Owner Matrix**
3. A mapping table appears under each file. Pick which column in *your* file feeds each field. Where a column in your file matches the usual name (see [Input Files](#input-files)) it is pre-selected, but every header in the file is available in the dropdown.
4. In Current Commits, tick **Vendor Code** and/or **Vendor Name** if you want commits matched to the Owner Matrix. Ticking one switches on its dropdown, and switches on the matching Owner Matrix column (see below).
5. Set the **Forecast Start Date**, the **Forecast Window** (1 to 90 days, default 14), and the **Commit Aggregation** (day, week, or month).
6. Click **Predict Risk**. The form will not submit, and says why, if a required column has not been chosen. The forecast is built and the forecast accuracy metrics appear (AVG Commit Qty, R², MAE, RMSE, MAPE). You can download the raw forecast as a CSV.
7. Click **Combine Forecast, Commits & Owner Matrix** to download `vendor_commit_risk.xlsx`.

Results are held in memory for the running server process, so the app is built for one user at a time.

---

## Input Files

Only the columns you map are read; other columns in your files are ignored. Column names in your files can be anything, because you map them in the UI.

### Historical Commits (required)

Past commits and what was actually delivered against them, one row per delivery.

| Field | What to map | Pre-selected if your file has |
|---|---|---|
| Material | Material / part number | `Material` |
| Vendor | Vendor **name** | `Name 1` |
| Commit Date | Date the vendor committed to deliver | `Date Due` |
| Commit Qty | Quantity committed | `Qty Due` |
| Actual Delivery Date | Date the delivery **actually arrived** | `Date Rcvd` |
| Actual Delivery Qty | Quantity that actually arrived | `Qty Rcvd` |

Days late is **not** mapped. It is calculated from Commit Date and Actual Delivery Date (see [`data_processing.py`](#data_processingpy---data-cleaning)). Rows missing either date are dropped.

**Mapping the OTD extract** (`New_Historical.OTD.xlsx`). Its column names don't match the pre-selected ones, so choose them yourself:

| Field | Column |
|---|---|
| Material | `Material` |
| Vendor | `VENDOR_NAME` |
| Commit Date | `Stat Date` |
| Commit Qty | `QUANTITY` |
| Actual Delivery Date | `Delivered in Full Date` (or `OTD_DT_SAT_IN_FULL_CORRECTED`, which is identical in this extract) |
| Actual Delivery Qty | `Received Qty` |

Do **not** use `Delivery Date` or `DELIVERY_DATE` as the Actual Delivery Date. Those include scheduled dates months after the last real receipt, which makes the accuracy metrics show N/A (see [Troubleshooting](#troubleshooting)).

**Keep the history in the past.** Historical Commits should only contain deliveries that happened *before* the Forecast Start Date. If it includes later deliveries, the model is trained on the same period it will be scored against.

### Current Commits (required)

The open commits to assess. Each row is a commit for a material on a date.

| Field | Required | Pre-selected if your file has | Description |
|---|---|---|---|
| Material | Yes | `Material` | Material / part number |
| Commit Date | Yes | `Commit Dt by Suppl` | Date the vendor committed to deliver |
| Commit Qty | Yes | `Commit Qty` | Quantity committed |
| Vendor Code | Optional (tick) | `Vendor` | Used **only** to match the Owner Matrix |
| Vendor Name | Optional (tick) | `Name 1` | Used to match the Owner Matrix, and as the vendor name in the output |

Vendor Code and Vendor Name are not read at all unless ticked. If neither is ticked, nothing is matched to the Owner Matrix and **Assigned Buyer is UNKNOWN for every row**. The Vendor Name column in the output then comes from the historical data.

### Owner Matrix (required)

Maps vendors to the buyer responsible for them.

| Field | Required | Pre-selected if your file has | Description |
|---|---|---|---|
| Assigned Buyer | Yes | `Assigned Buyer` | Buyer responsible for the vendor |
| Vendor Code (Supplier #) | Only if Vendor Code is ticked | `Supplier #` | Matched to the commit's Vendor Code |
| Vendor Name (Supplier Name) | Only if Vendor Name is ticked | `Supplier Name` | Matched to the commit's Vendor Name |

**How matching works:** vendor codes are matched first (trimmed and leading zeros removed; commit codes are also upper-cased, so keep the Owner Matrix codes in upper case). Commits still unmatched are then matched on vendor name (upper-cased, extra spaces ignored). Anything still unmatched is UNKNOWN. A blank code or name never matches.

---

## File-by-File Breakdown

### `app.py` - The Front Door
Flask web app. Routes: `/` (upload form), `/get_columns` (reads a file's headers for the column-mapping table), `/run_forecast` (checks the uploads and mappings, remaps the files, runs `upload_services`, stores the results), `/download_forecast.csv`, and `/combine` (runs `combine_services` and writes the Excel file with the **Risk Output** and **Accuracy** sheets).

Each uploaded file is rewritten to hold only the mapped fields, named the way the rest of the app expects. Required fields are checked in the browser (a message appears and the form doesn't submit) and again on the server (HTTP 400 with the same message). The server starts on port 5000 with `debug=True` and `host="0.0.0.0"`.

### `upload_services.py` - Ingestion and Forecast Orchestration
Loads Historical Commits, calls `build_forecast`, then reads the Current Commits and Owner Matrix files. For Current Commits it trims and upper-cases Material (and Vendor Code, if ticked) and renames `Commit Date` to `Commit Dt by Suppl` and `Vendor Code` to `Vendor` for the combine step. Returns the forecast, commits, and owner matrix as CSV bytes plus the accuracy metrics. It does **not** write the final Excel file (that happens in `app.py`).

The file still contains older code for converting an OTD upload. The UI no longer offers that upload, so it is not used.

### `data_processing.py` - Data Cleaning
Cleans the Historical Commits file:
- Keeps the six mapped fields and renames them for the rest of the app: `Vendor` to `Vendor Name`, `Commit Date` to `Date Due`, `Commit Qty` to `Quantity Due`, `Actual Delivery Date` to `Date Received`, `Actual Delivery Qty` to `Quantity Received`
- Normalizes material numbers (takes the first token and strips trailing letters)
- Fixes date and number formats
- **Calculates `Number of Days Late`** as working days (Mon-Fri) from Commit Date to Actual Delivery Date. Weekends are skipped, and so are the days listed in `non_working_days()`: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and the day after, and Dec 24-31 (weekend holidays are taken on the nearest weekday). Early deliveries count as on time (0). Working days are used because the ERP's own "Days Late" column is measured that way. On the on-time and late rows of the 12/31/25 extract this gives the same number as that column about 96% of the time and the same lateness band 99.7% of the time. Edit `non_working_days()` to match your own plant calendar
- Derives `Days Late Classification`: On Time, 1 Day Late, 2-4 Days, 5-15 Days, >15 Days
- Sets `Number of Days Late` to 0 when `Quantity Received >= Quantity Due` (over-delivery is not a failure). This happens *after* the classification is derived, so an over-delivered but late row keeps its late classification. In the 12/31/25 extract 99.7% of rows are delivered in full, so days late (and therefore `Avg Days Late`) is close to 0 and lateness reaches the model mainly through the classification
- Drops rows with no commit date or no actual delivery date
- Adds `Avg Days Late`: each vendor's mean days late across all of their rows

### `model.py` - The Forecast Model
Trains an XGBoost regressor (50 trees, depth 3, learning rate 0.1, seed 42) on one Material + Vendor pair's delivery history. The target is `Quantity Received`. Features:

| Feature | In training | At forecast time |
|---|---|---|
| Trend | Days since the first delivery | Days since the first delivery (extends past the training range) |
| Delivery description score | Lateness score of *that same delivery*: on time 0, 1 day late 1, 2-4 days 3, 5-15 days 10, >15 days 20 | Set to the historical average |
| Vendor performance | Vendor's `Avg Days Late` | Set to the historical average |
| Seasonality | Month of the delivery | Month of the forecast date |

**Output:** an array of forecast quantities (`mu`, floored at 0), one per day of the window, plus a single `sigma` (standard deviation of the training residuals). Because the description score and vendor value are constants at forecast time, the forecast varies across the window only through the date trend and month.

Commit Date and Commit Qty from Historical Commits are **not** model inputs. They only feed the lateness calculation and the over-delivery rule above.

### `forecast_build_services.py` - Forecast Runner
Builds the list of future dates, runs the accuracy backtest, and loops through every Material + Vendor pair calling `model.py`. Pairs with fewer than 5 historical rows (or any other error) are skipped silently. Output rows are labeled with the *latest* vendor seen for the material.

### `forecast_accuracy_services.py` - Forecast Accuracy Tracker
A holdout backtest: the last `horizon` days of history (by Actual Delivery Date) are held out, the model is trained on everything earlier (per material), and its forecasts are compared with the deliveries that actually arrived on matching material/date rows. Reports:

- `AVG_COMMIT_QTY` (mean actual quantity of the matched rows), `MAE`, `RMSE`, `MAPE`, `R2`

MAE, RMSE, and MAPE only count under-prediction (shortfalls); over-prediction counts as zero error. The result is one set of numbers for the whole run, not per vendor. If nothing can be compared, all values are "N/A" (see [Troubleshooting](#troubleshooting)).

### `combine_services.py` - The Assembler
- Filters commits and forecasts to the forecast window and rolls dates up by the chosen aggregation: **day** (no change), **week** (coming Sunday), **month** (month end)
- Commit quantities are summed per Material + bucket
- For week and month buckets, the forecast is taken from the **single latest forecast day in the bucket** (daily forecasts are not summed)
- Matches commits to forecasts on Material + date; if several forecasts share a material and date, the one with the lowest sigma is kept
- Assigns the buyer from the Owner Matrix: by Vendor Code first, then by Vendor Name, only for the fields the user ticked. Otherwise UNKNOWN
- Vendor Name in the output: the Current Commits Vendor Name if ticked; otherwise the latest vendor seen for that material in the historical data; then the Owner Matrix name (matched by code), then the vendor code
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
2. A receipts file with `Material`, `Document Date`, and `Quantity` columns (a different format from the Historical Commits file)

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

Results from saved evaluation runs in `Data/Performance/`. Runs use different data and code versions, and these predate the Historical Commits change (days late is now calculated from dates), so re-run `performance_check.py` after any change.

| Run file | Records | Classes | Accuracy | HIGH precision | HIGH recall | Share of rows actually HIGH |
|---|---|---|---|---|---|---|
| `4.27.risk_output_with_actuals.xlsx` | 428 | 3 (HIGH/MED/LOW) | 33.2% | 37.6% | 66.3% | 39.5% |
| `risk_output_with_actuals.99.xlsx` | 456 | 2 | 41.5% | 39.6% | 65.3% | 43.6% |
| `risk_output_with_actuals.z.xlsx` | 349 | 2 | 47.0% | 20.9% | 9.6% | 41.8% |

**How to read this:** precision is the share of HIGH-flagged commits that turned out to be HIGH. In every run above it is *below* the overall share of HIGH outcomes, so a HIGH flag does not currently identify riskier commits than an average commit. The first run's 3-class accuracy is not comparable with the later 2-class runs (MED was removed from the risk labels). Until this improves, the labels should not be used to prioritize vendors.

In the first two runs the model flagged about 70% of commits HIGH while roughly 40-44% were actually HIGH, so accuracy alone can look similar across very different inputs. Compare precision against the share of rows actually HIGH, not just accuracy, when judging a change.

---

## Troubleshooting

**Forecast Accuracy shows N/A.** The accuracy backtest holds out the last *N* days (the Forecast Window) of Actual Delivery Date, and N/A means nothing could be compared. Common causes:

- **The Actual Delivery Date column contains dates beyond the real receipts**, such as scheduled delivery dates. The latest date sets the holdout window, so one date months in the future moves the window past all your data. Map a true receipt-date column instead.
- **The holdout materials have fewer than 5 earlier rows**, so no forecast could be built for them.
- **The Forecast Window is very short and few deliveries fall inside it.** A longer window holds out more rows.

**A commit is missing from the output.** Commits with no matching forecast are dropped. This happens when the material has no Material + Vendor pair with 5 or more historical rows, when the material numbers don't match after cleaning, or when the commit date is outside the forecast window.

**Assigned Buyer is UNKNOWN.** Either neither Vendor Code nor Vendor Name was ticked in Current Commits, or the code/name in the commit file doesn't match the Owner Matrix.

---

## Known Limitations

**Modeling**
- The model forecasts delivery **quantity**. Nothing in it predicts whether a delivery arrives by the commit date, but `performance_check.py` scores quantity received by the commit date, so timing affects the scoring and not the prediction.
- The vendor performance feature is a per-vendor average, and each model is trained on a single Material + Vendor pair, so it is the same value on every training row and carries no signal.
- The delivery description score is the lateness of the same delivery whose quantity is being predicted, not earlier history, and it is replaced by the average at forecast time.
- `sigma` is measured on the training data, so it underestimates true forecast error, and it is the same for every day of the window. A walk-forward (out-of-sample) sigma is planned.
- Tree models cannot extrapolate the date trend beyond the training range.
- A Material + Vendor pair needs at least 5 historical rows; otherwise it is skipped with no message.
- The model is retrained on every run (and again for the accuracy backtest). No model file is saved.

**Risk engine**
- Missing inputs are scored 0% / LOW, which reads as a confident "low risk" rather than "unknown".
- `sigma = 0` forces 50% probability and therefore HIGH.
- The confidence cutoffs are fixed constants (see above).

**Data handling**
- Days late is calculated from the two dates, using an approximate holiday calendar. It matches the ERP column closely but not exactly.
- For materials with more than one vendor, forecasts are matched on Material + date only (lowest sigma wins), and forecast rows are labeled with the latest vendor. This can attach one vendor's forecast to another vendor's commit.
- Material numbers are normalized differently in Historical Commits (trailing letters stripped) and Current Commits (trimmed and upper-cased only). Materials that don't match after these steps get no forecast.
- Commits with no matching forecast are dropped from the output.
- The accuracy metrics ignore over-prediction and are computed once per run.
- Nothing stops Historical Commits from containing deliveries after the Forecast Start Date (see [Input Files](#input-files)).

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
| `templates/index.html` | Web UI (file upload and column mapping) |
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