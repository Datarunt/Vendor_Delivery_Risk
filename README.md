# Vendor Delivery Risk Predictor

A Python/Flask tool that looks at each open vendor commit (a date and a quantity) and labels it **HIGH** or **LOW** risk of being missed, before the commit date arrives. A commit is "missed" when the delivery is late or short.

The label comes from a machine-learned model trained on past commits and what actually arrived against them. It uses each vendor's and material's delivery record, how big the commit is, and the time of year. An XGBoost quantity forecast is still built alongside it and shown in the output, but it no longer decides the label.

> **Status: under active development.** On a test set of 456 commits the model's labels were right 64% of the time, up from 41% for the previous quantity-based logic (see [Model Performance](#model-performance)). The test set is small (about 37 vendors, one time window) and the cutoff was tuned on it, so treat the output as experimental and check it on new commits before relying on it.

---

## What Problem Does This Solve?

Vendors make delivery commitments (a date and a quantity). The question this tool tries to answer is:

> **"Given how this vendor has delivered on this material in the past, how likely is it that this commit is met, on time and in full?"**

The goal is an early warning so the team can follow up, find alternatives, or adjust plans before a shortfall happens.

---

## How It Works

In plain terms:

1. **Historical Commits** (past commits and what actually arrived against them) are cleaned. For every past delivery the tool calculates how many working days late it was.
2. **A quantity forecast is built** for each Material + Vendor pair with at least 5 past deliveries: an XGBoost regressor forecasts the quantity that will arrive on each day of the forecast window, with an uncertainty range. It feeds the quantity columns in the output, and decides which commits appear (a commit needs a quantity forecast to be listed).
3. **A lateness scorecard** works out how often each vendor, and each vendor on each material, has delivered late. Thin records are blended toward broader averages. This gives an on-time probability.
4. **A miss model** (an XGBoost classifier) is trained on every past commit, with "missed" (late or short) as the outcome. For each past commit its inputs are what was knowable while that commit was still open: the vendor's and material's record up to that point, the commit's size compared with what is usual for that material, and the month and weekday. It gives each current commit a chance of being met.
5. **The Risk label** is HIGH when the model's chance of being met is below 85%. If the model can't be trained, the scorecard is used, and if that isn't available, the quantity forecast.
6. Optionally, each commit is matched to the **Owner Matrix** by vendor code and/or vendor name to add the Assigned Buyer.

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
forecast + history rows (in memory, downloadable as CSV) + accuracy metrics shown in the UI
   |
   |  "Combine" button
   v
combine_services.py           <- matches commits to forecasts and to the Owner Matrix
   |-- lateness_services.py   <- late-rate scorecard  -> On-Time Probability
   |-- ml_risk_services.py    <- XGBoost miss model   -> Model On-Time Probability
   '-- risk_engine.py         <- Risk label (model, then scorecard, then quantity),
   |                             plus quantity probability, interval, confidence
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
6. Click **Predict Risk**. The form will not submit, and says why, if a required column has not been chosen. The quantity forecast is built and its accuracy metrics appear (AVG Commit Qty, R², MAE, RMSE, MAPE). You can download the raw forecast as a CSV.
7. Click **Combine Forecast, Commits & Owner Matrix** to download `vendor_commit_risk.xlsx`. Combine uses the forecast from the last Predict Risk, so click Predict Risk again after changing any file or setting.

While it runs, the console should print lines like these. If it prints a warning instead, see [Troubleshooting](#troubleshooting):

```
Lateness history: 96329 deliveries before 2026-04-27, 213 vendors, 3559 Material + Vendor pairs. Overall late rate 19.2%.
Miss model: trained on 96258 past commits (missed 19.2%), inputs as of 163 days before each commit.
```

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

All six fields are used. Commit Date and Actual Delivery Date give each delivery's lateness, and Commit Qty and Actual Delivery Qty show whether it arrived short. The miss model needs at least 500 usable past commits; with fewer it is skipped and the scorecard is used instead.

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

**Keep the history in the past.** Historical Commits should only contain deliveries that happened *before* the Forecast Start Date. The lateness scorecard and the miss model ignore deliveries on or after that date automatically, so a commit is never judged with deliveries that came after it. The quantity forecast does not, and would be trained on the same period it is compared with.

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

The vendor used for the lateness scorecard and the miss model is always the one seen for that material in the historical data, whichever of these boxes are ticked.

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
- **Calculates `Working Days Late`** as working days (Mon-Fri) from Commit Date to Actual Delivery Date. Weekends are skipped, and so are the days listed in `non_working_days()`: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and the day after, and Dec 24-31 (weekend holidays are taken on the nearest weekday). Early deliveries count as on time (0). Working days are used because the ERP's own "Days Late" column is measured that way. On the on-time and late rows of the 12/31/25 extract this gives the same number as that column about 96% of the time and the same lateness band 99.7% of the time. Edit `non_working_days()` to match your own plant calendar
- Stores that value in `Working Days Late` and leaves it untouched. The lateness scorecard and the miss model use this column
- Derives `Days Late Classification`: On Time, 1 Day Late, 2-4 Days, 5-15 Days, >15 Days
- Starts `Number of Days Late` at the same value, then sets it to 0 when `Quantity Received >= Quantity Due` (over-delivery is not a failure). This happens *after* the classification is derived, so an over-delivered but late row keeps its late classification. In the 12/31/25 extract 99.7% of rows are delivered in full, so `Number of Days Late` (and therefore `Avg Days Late`) is close to 0. That is why lateness is read from `Working Days Late` instead
- Drops rows with no commit date or no actual delivery date
- Adds `Avg Days Late`: each vendor's mean days late across all of their rows (used by the quantity forecast)

### `model.py` - The Quantity Forecast
Trains an XGBoost regressor (50 trees, depth 3, learning rate 0.1, seed 42) on one Material + Vendor pair's delivery history. The target is `Quantity Received`. Features:

| Feature | In training | At forecast time |
|---|---|---|
| Trend | Days since the first delivery | Days since the first delivery (extends past the training range) |
| Delivery description score | Lateness score of *that same delivery*: on time 0, 1 day late 1, 2-4 days 3, 5-15 days 10, >15 days 20 | Set to the historical average |
| Vendor performance | Vendor's `Avg Days Late` | Set to the historical average |
| Seasonality | Month of the delivery | Month of the forecast date |

**Output:** an array of forecast quantities (`mu`, floored at 0), one per day of the window, plus a single `sigma` (standard deviation of the training residuals). Because the description score and vendor value are constants at forecast time, the forecast varies across the window only through the date trend and month.

This forecast no longer decides the Risk label. It feeds the `Quantity Probability`, `Confidence Interval` and `Confidence` columns, and it decides which commits are listed (a commit needs a forecast to appear).

### `forecast_build_services.py` - Forecast Runner
Builds the list of future dates, runs the accuracy backtest, and loops through every Material + Vendor pair calling `model.py`. Pairs with fewer than 5 historical rows (or any other error) are skipped silently. Output rows are labeled with the *latest* vendor seen for the material. The result holds both the forecast rows and the history rows, and the history rows carry `Working Days Late` on to the combine step.

### `forecast_accuracy_services.py` - Forecast Accuracy Tracker
A holdout backtest of the **quantity forecast**: the last `horizon` days of history (by Actual Delivery Date) are held out, the model is trained on everything earlier (per material), and its forecasts are compared with the deliveries that actually arrived on matching material/date rows. Reports:

- `AVG_COMMIT_QTY` (mean actual quantity of the matched rows, despite the name), `MAE`, `RMSE`, `MAPE`, `R2`

MAE, RMSE, and MAPE only count under-prediction (shortfalls); over-prediction counts as zero error, so they look better than a two-sided error would. R2 counts errors in both directions. RMSE is much larger than MAE when a few high-volume rows are badly under-predicted. The result is one set of numbers for the whole run, not per vendor. If nothing can be compared, all values are "N/A" (see [Troubleshooting](#troubleshooting)). These numbers describe the quantity forecast only and do not measure the Risk label. Use `performance_check.py` for that.

### `lateness_services.py` - The Lateness Scorecard
Works out how often deliveries are late, and turns that into an on-time probability for each commit.

- A delivery counts as **late** when it arrives `LATE_AFTER_WORKING_DAYS` (1) or more working days after its commit date. This matches how commits are scored: at the commit date, with no grace period
- Only deliveries received **before the Forecast Start Date** are used
- Late rates are worked out for the whole history, for each vendor, and for each Material + Vendor pair. Thin records are pulled toward the broader rate: a pair toward its vendor, and a vendor toward the overall rate. The pull is worth `PRIOR_STRENGTH` (20) deliveries, so a pair with 20 deliveries counts its own record and its vendor's equally, and a pair with 2 is almost entirely its vendor's rate. This stops a material with 3 on-time deliveries from looking 100% reliable
- **On-Time Probability = 1 - the late rate.** A pair with no history gets its vendor's rate, and an unknown vendor gets the overall rate
- `On-Time Sample Size` reports how many deliveries were behind the pair's own record (0 means the estimate is entirely the vendor's or the overall rate)

### `ml_risk_services.py` - The Miss Model
An XGBoost classifier that predicts whether a commit will be **missed** (late, or short of the committed quantity). It learns from the Historical Commits rows, so it is trained on every run.

- **Training examples:** every past commit received before the Forecast Start Date. The outcome is missed (1) if it arrived `LATE_AFTER_WORKING_DAYS` or more working days late, or arrived short; otherwise 0
- **Inputs** (12), all known while the commit was still open:
  - the commit itself: its quantity, its size compared with this material's usual commit size, the month and the weekday
  - the vendor's record: late rate, average days late, share of severely late deliveries (5 or more working days), and late rate over the last 180 days
  - the vendor on this material: late rate, average days late, and late rate over the last 180 days
  - the scorecard's late-rate estimate, so the model starts from it and only has to learn what it misses
- **No peeking:** a commit's inputs use only deliveries received *before* that commit's as-of date, never its own outcome or anything later. For training, as-of is the commit date minus a lead time. The lead time is the typical gap between the newest delivery on record and the commit dates being scored (0 to 365 days; 30 if it can't be measured), so the model is trained on inputs as out of date as the ones it is given. Commits already delivered before their as-of date are left out of training, since they would no longer have been open
- **Not inputs on purpose:** the number of deliveries a vendor or material has on record. Counts grow with calendar time, so the model used them as a stand-in for "which era is this" and learned a pattern that reversed on newer commits. With them it did worse than chance on the test data
- **Prediction:** the chance a commit is met, `1 - P(missed)`, kept between 0.1% and 99.9%. For week or month aggregation the model is given the average commit size in the bucket, since it was trained on single commits
- **Skipped** (with a console warning) if there are fewer than 500 usable past commits or the history lacks the columns it needs. The scorecard is then used

The model is retrained on every run and is not saved between sessions. It is only trained when `RISK_BASIS` is `"model"`.

### `combine_services.py` - The Assembler
- Filters commits and forecasts to the forecast window and rolls dates up by the chosen aggregation: **day** (no change), **week** (coming Sunday), **month** (month end)
- Commit quantities are summed per Material + bucket
- For week and month buckets, the quantity forecast is taken from the **single latest forecast day in the bucket** (daily forecasts are not summed)
- Matches commits to forecasts on Material + date; if several forecasts share a material and date, the one with the lowest sigma is kept
- Builds the lateness scorecard and trains the miss model from the history rows, then adds `On-Time Probability`, `On-Time Sample Size` and `Model On-Time Probability` to each commit
- Assigns the buyer from the Owner Matrix: by Vendor Code first, then by Vendor Name, only for the fields the user ticked. Otherwise UNKNOWN
- Vendor Name in the output: the Current Commits Vendor Name if ticked; otherwise the latest vendor seen for that material in the historical data; then the Owner Matrix name (matched by code), then the vendor code
- Calls `risk_engine.py` and returns the final table
- **Commits with no matching quantity forecast are dropped** from the output (a warning is printed to the console)

### `risk_engine.py` - The Risk Scorer
Decides the **Risk** and **Probability** columns, and produces the quantity results.

**What Risk is based on** (`RISK_BASIS`, default `"model"`):

| Basis | Probability used | HIGH when it is below |
|---|---|---|
| `"model"` | `Model On-Time Probability` from the miss model | 85% (`MODEL_HIGH_RISK_BELOW`) |
| `"lateness"` | `On-Time Probability` from the scorecard | 90% (`LATENESS_HIGH_RISK_BELOW`) |
| `"quantity"` | `Quantity Probability` from the quantity forecast | 75% |

Each basis falls back to the next when a row has no value: model, then lateness, then quantity. The probability is rounded to a whole percent before it is compared with the cutoff. The model's cutoff is lower than the scorecard's because its probabilities are spread wider and sit lower (it also counts short deliveries as misses).

**Quantity results** (always produced): `Quantity Probability = P(Normal(mu, sigma) >= Commit Qty)`. The interval is `mu ± 1.96 × sigma`, floored at 0. A separate **Confidence** level is based on the interval's relative width, `(upper - lower) / mu`:

| Relative width | Confidence |
|---|---|
| 0.0356 or less | HIGH |
| Up to 0.1227 | MED |
| Above 0.1227 | LOW |

These two cutoffs are hard-coded constants derived from the tertiles of one dataset. `Confidence Interval` and `Confidence` describe the **quantity forecast**, not the Risk label.

Special cases for the quantity probability: if `mu`, `sigma`, or `Commit Qty` is missing, the row is scored 0% / LOW. If `sigma` is 0, probability is set to 50%, which is HIGH.

### `performance_check.py` - Model Scorecard
Run separately (`python performance_check.py`) after deliveries have happened. File dialogs ask for:

1. The prediction file (`vendor_commit_risk.xlsx`, read from its **Risk Output** sheet)
2. A receipts file with `Material`, `Document Date`, and `Quantity` columns (a different format from the Historical Commits file)

For each commit row, receipts are counted up to the commit date, with each period starting after the previous commit date for that material. Surplus and shortfall carry forward to the next period. Then:

- `Cum_Actual` and `Cum_Commit` are the received and committed quantities for the period, after carry-over
- `Fulfillment_Ratio = Cum_Actual / Cum_Commit`
- `Actual_Risk = LOW if the ratio is 1.0 or more, otherwise HIGH`
- `Performance` is CORRECT when the predicted `Risk` matches `Actual_Risk`

A commit filled only after its commit date has a low ratio, so it counts as HIGH. The output also carries `Has_Late_Delivery`, which is TRUE when the receipts file has any receipt for the material dated after the commit date. It is informational: it does not feed `Actual_Risk` or any metric, and can be deleted from the script.

Only commits dated inside the receipts file's date range are scored. **Output:** an Excel file with **Risk Output** (with actuals appended), **Confusion Matrix** (HIGH/LOW), **Metrics** (records, accuracy, HIGH precision, recall, F1), and **Classification Report**.

### `vendor-commit-eda.py` - Exploratory Analysis (optional)
A standalone script. It opens a file picker and a column-selection window, then writes a Word (`.docx`) report of tables and control charts. It is not used by the app.

---

## Output

`vendor_commit_risk.xlsx`:

- **Risk Output:**

| Column | Meaning |
|---|---|
| Material, Vendor Name, Vendor Commit, Commit Qty | The commit |
| Probability | The chance the commit is met, from whichever basis produced the Risk (`RISK_BASIS`) |
| Risk | HIGH or LOW |
| Confidence Interval, Confidence | Range and confidence of the **quantity forecast** |
| Assigned Buyer | From the Owner Matrix, or UNKNOWN |
| Model On-Time Probability | The miss model's chance the commit is met |
| On-Time Probability | The scorecard's chance the commit is on time |
| Quantity Probability | The quantity forecast's chance of reaching the commit quantity |
| On-Time Sample Size | Deliveries behind this Material + Vendor pair's own record (0 = based on the vendor or overall rate) |

- **Accuracy:** the backtest metrics of the quantity forecast, from `forecast_accuracy_services.py`

---

## Settings You Can Change

These are constants at the top of each file.

| Setting | File | Default | What it does |
|---|---|---|---|
| `RISK_BASIS` | `risk_engine.py` | `"model"` | Which probability decides Risk: `"model"`, `"lateness"` or `"quantity"` |
| `MODEL_HIGH_RISK_BELOW` | `risk_engine.py` | 0.85 | HIGH when the model's chance of being met is below this |
| `LATENESS_HIGH_RISK_BELOW` | `risk_engine.py` | 0.90 | The same cutoff for the scorecard |
| `LATE_AFTER_WORKING_DAYS` | `lateness_services.py` | 1 | Working days after the commit date before a delivery counts as late (also the definition of a miss for the model) |
| `PRIOR_STRENGTH` | `lateness_services.py` | 20 | How strongly thin records are pulled toward broader averages |
| `RECENT_DAYS` | `ml_risk_services.py` | 180 | Look-back for the "recent" inputs |
| `SEVERE_LATE_DAYS` | `ml_risk_services.py` | 5 | Working days late that counts as severely late |
| `DEFAULT_LEAD_DAYS` | `ml_risk_services.py` | 30 | Lead time used when it can't be measured |
| `MIN_TRAINING_ROWS` | `ml_risk_services.py` | 500 | Fewest usable past commits needed to train the model |
| `MODEL_PARAMS` | `ml_risk_services.py` | 200 trees, depth 3 | XGBoost settings for the miss model |
| `non_working_days()` | `data_processing.py` | US holidays + Dec 24-31 | The holiday calendar used for working days |

The two risk cutoffs were chosen on one set of test commits. Re-tune them with `performance_check.py` whenever the data changes materially: raising a cutoff flags more commits HIGH.

---

## Model Performance

Results on the 456 commits in the 4/27 snapshot that fall inside the receipts file (`4.27to6.9`), with the forecast start on 4/27/2026. 199 of the 456 (43.6%) were actually HIGH.

| Risk based on | Accuracy | HIGH precision | HIGH recall |
|---|---|---|---|
| Quantity forecast (previous logic) | 41.4% | 39.6% | 65.3% |
| Lateness scorecard | 61.2% | 56.4% | 48.7% |
| Miss model (current default) | 64.0% | 58.2% | 62.3% |

For the current default the full report is: HIGH precision 58.2%, recall 62.3%, F1 60.2%; LOW precision 69.1%, recall 65.4%, F1 67.2%. Roughly 47% of commits are flagged HIGH.

**How to read this:** precision is the share of HIGH-flagged commits that turned out to be HIGH. The previous logic's precision (39.6%) was below the share of commits that were actually HIGH (43.6%), so its HIGH flag told you nothing. The current model's (58.2%) is well above it. The previous logic also called almost everything HIGH: it recognised only 23% of the commits that turned out fine, against 65% now. A month-by-month test through 2025 on the historical data, scoring commits that were still open at the start of each month, also had the model ahead of the scorecard in 10 of 11 months.

**Caution:**
- The test set is small: 456 commits from about 37 vendors, in one time window.
- The 85% and 90% cutoffs were chosen after looking at these same commits, so the figures are probably a little optimistic.
- The real test is a new batch of commits, scored without changing anything.

Earlier saved runs in `Data/Performance/` used the previous quantity-based logic.

---

## Troubleshooting

**Forecast Accuracy shows N/A.** The accuracy backtest holds out the last *N* days (the Forecast Window) of Actual Delivery Date, and N/A means nothing could be compared. Common causes:

- **The Actual Delivery Date column contains dates beyond the real receipts**, such as scheduled delivery dates. The latest date sets the holdout window, so one date months in the future moves the window past all your data. Map a true receipt-date column instead.
- **The holdout materials have fewer than 5 earlier rows**, so no forecast could be built for them.
- **The Forecast Window is very short and few deliveries fall inside it.** A longer window holds out more rows.

**The results look the same as before the lateness update, and Risk Output has no On-Time columns.** If `Probability` equals `Quantity Probability` on every row and there are no `On-Time Probability` or `Model On-Time Probability` columns, the app has fallen back to the quantity forecast. The console shows `WARNING: 'Working Days Late' is not in the forecast data`. This means the project's `data_processing.py` is an older version that does not create the `Working Days Late` column. Replace it with the current one, restart the app, and click Predict Risk again before Combine. Also check that `lateness_services.py` and `ml_risk_services.py` are in the project folder.

**The console says the miss model was skipped.** Either there were fewer than 500 usable past commits, or the history is missing columns the model needs (`Date Due`, `Quantity Due` and so on, which come from the Historical Commits mapping). Risk then comes from the lateness scorecard.

**A commit is missing from the output.** Commits with no matching quantity forecast are dropped. This happens when the material has no Material + Vendor pair with 5 or more historical rows, when the material numbers don't match after cleaning, or when the commit date is outside the forecast window.

**Assigned Buyer is UNKNOWN.** Either neither Vendor Code nor Vendor Name was ticked in Current Commits, or the code/name in the commit file doesn't match the Owner Matrix.

---

## Known Limitations

**The risk label**
- It predicts whether a commit will be missed, from the vendor's and material's delivery record. It does not look at the specific order, open purchase orders, lead times or anything else about that commit.
- Lateness in the history is measured against the historical Commit Date, while current commits use `Commit Dt by Suppl`. The two are assumed to mean the same thing. If a vendor's commit date is usually later than the historical due date, the history will overstate lateness.
- Shortfalls count as misses, but they are rare in the history (297 of 96,586 rows in the 12/31/25 extract), so the model is mostly learning lateness.
- `performance_check.py` scores a commit by cumulative quantity received by the commit date, with carry-over between commits. The model predicts a single commit's outcome, so the two don't always line up. In the test set about a quarter of the commits that were actually HIGH were partial receipts rather than fully late.
- The risk cutoffs were tuned on one small test set (see [Model Performance](#model-performance)).
- A pair with no history gets its vendor's rate, and an unknown vendor the overall rate (`On-Time Sample Size` is 0 for these).
- For materials with more than one vendor, the lateness scorecard and the miss model use the latest vendor seen for the material.
- The miss model is retrained on every run and is not saved.

**The quantity forecast** (no longer decides Risk)
- The vendor performance feature is a per-vendor average, and each model is trained on a single Material + Vendor pair, so it is the same value on every training row and carries no signal.
- The delivery description score is the lateness of the same delivery whose quantity is being predicted, not earlier history, and it is replaced by the average at forecast time.
- `sigma` is measured on the training data, so it underestimates true forecast error, and it is the same for every day of the window. A walk-forward (out-of-sample) sigma is planned.
- Tree models cannot extrapolate the date trend beyond the training range.
- A Material + Vendor pair needs at least 5 historical rows; otherwise it is skipped with no message, and its commits are dropped from the output.
- It still trains on every historical row, including any dated after the Forecast Start Date.
- The quantity probability scores missing inputs as 0% / LOW, and `sigma = 0` as 50% (HIGH). The confidence cutoffs are fixed constants.
- The accuracy metrics ignore over-prediction and are computed once per run.

**Data handling**
- Days late is calculated from the two dates, using an approximate holiday calendar. It matches the ERP column closely but not exactly.
- For materials with more than one vendor, quantity forecasts are matched on Material + date only (lowest sigma wins), and forecast rows are labeled with the latest vendor. This can attach one vendor's forecast to another vendor's commit.
- Material numbers are normalized differently in Historical Commits (trailing letters stripped) and Current Commits (trimmed and upper-cased only). Materials that don't match after these steps get no forecast.

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
| `lateness_services.py`, `ml_risk_services.py` | The lateness scorecard and the miss model |
| `performance_check.py` | Scores past predictions against actual receipts |
| `vendor-commit-eda.py` | Exploratory analysis script |
| `templates/index.html` | Web UI (file upload and column mapping) |
| `Data/` | Working files: Historical, Commits, Account Matrix, Actuals, Predictions, Performance, EDA |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Quantity forecast (regressor) and miss model (classifier) |
| SciPy | Normal-distribution probability calculation |
| Pandas / NumPy | Data processing |
| Flask | Web interface |
| Scikit-learn | Evaluation metrics (`performance_check.py`) |
| XlsxWriter / openpyxl | Excel output and input |
| matplotlib / python-docx | EDA report (optional) |