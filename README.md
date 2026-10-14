--- src/vendor_commit_risk/README.md	2026-09-20 18:41:00.000000000 -0700
+++ out/README.md	2026-10-14 12:15:42.590490908 -0700
@@ -145,7 +145,7 @@
 | Vendor Code (Supplier #) | Only if Vendor Code is ticked | `Supplier #` | Matched to the commit's Vendor Code |
 | Vendor Name (Supplier Name) | Only if Vendor Name is ticked | `Supplier Name` | Matched to the commit's Vendor Name |
 
-**How matching works:** vendor codes are matched first (trimmed and leading zeros removed; commit codes are also upper-cased, so keep the Owner Matrix codes in upper case). Commits still unmatched are then matched on vendor name (upper-cased, extra spaces ignored). Anything still unmatched is UNKNOWN. A blank code or name never matches.
+**How matching works:** vendor codes are matched first (trimmed, any `.0` removed so a code read as `12345.0` becomes `12345`, and leading zeros removed; commit codes are also upper-cased, so keep the Owner Matrix codes in upper case). Commits still unmatched are then matched on vendor name (upper-cased, extra spaces ignored). Anything still unmatched is UNKNOWN. A blank code or name never matches.
 
 ---
 
@@ -168,7 +168,7 @@
 - Fixes date and number formats
 - **Calculates `Working Days Late`** as working days (Mon-Fri) from Commit Date to Actual Delivery Date. Weekends are skipped, and so are the days listed in `non_working_days()`: New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and the day after, and Dec 24-31 (weekend holidays are taken on the nearest weekday). Early deliveries count as on time (0). Working days are used because the ERP's own "Days Late" column is measured that way. On the on-time and late rows of the 12/31/25 extract this gives the same number as that column about 96% of the time and the same lateness band 99.7% of the time. Edit `non_working_days()` to match your own plant calendar
 - Stores that value in `Working Days Late` and leaves it untouched. The lateness scorecard and the miss model use this column
-- Derives `Days Late Classification`: On Time, 1 Day Late, 2-4 Days, 5-15 Days, >15 Days
+- Derives `Days Late Classification`: On Time, 1 Day Late, 2-4 Days, 5-15 Days, >15 Days. The labels in the data say "Calendar Days" (for example `Miss / 2-4 Calendar Days`), carried over from the ERP band names, but the bands are counted in working days
 - Starts `Number of Days Late` at the same value, then sets it to 0 when `Quantity Received >= Quantity Due` (over-delivery is not a failure). This happens *after* the classification is derived, so an over-delivered but late row keeps its late classification. In the 12/31/25 extract 99.7% of rows are delivered in full, so `Number of Days Late` (and therefore `Avg Days Late`) is close to 0. That is why lateness is read from `Working Days Late` instead
 - Drops rows with no commit date or no actual delivery date
 - Adds `Avg Days Late`: each vendor's mean days late across all of their rows (used by the quantity forecast)
@@ -188,7 +188,7 @@
 This forecast no longer decides the Risk label. It feeds the `Quantity Probability`, `Confidence Interval` and `Confidence` columns, and it decides which commits are listed (a commit needs a forecast to appear).
 
 ### `forecast_build_services.py` - Forecast Runner
-Builds the list of future dates, runs the accuracy backtest, and loops through every Material + Vendor pair calling `model.py`. Pairs with fewer than 5 historical rows (or any other error) are skipped silently. Output rows are labeled with the *latest* vendor seen for the material. The result holds both the forecast rows and the history rows, and the history rows carry `Working Days Late` on to the combine step.
+Builds the list of future dates, runs the accuracy backtest, and loops through every Material + Vendor pair calling `model.py`. Pairs with fewer than 5 historical rows (or any other error) are skipped. A `FORECAST ERROR` line naming the material and vendor is printed to the console, but nothing is shown in the browser. Output rows are labeled with the *latest* vendor seen for the material. The result holds both the forecast rows and the history rows, and the history rows carry `Working Days Late` on to the combine step.
 
 ### `forecast_accuracy_services.py` - Forecast Accuracy Tracker
 A holdout backtest of the **quantity forecast**: the last `horizon` days of history (by Actual Delivery Date) are held out, the model is trained on everything earlier (per material), and its forecasts are compared with the deliveries that actually arrived on matching material/date rows. Reports:
@@ -201,7 +201,7 @@
 Works out how often deliveries are late, and turns that into an on-time probability for each commit.
 
 - A delivery counts as **late** when it arrives `LATE_AFTER_WORKING_DAYS` (1) or more working days after its commit date. This matches how commits are scored: at the commit date, with no grace period
-- Only deliveries received **before the Forecast Start Date** are used
+- Only deliveries received **before the Forecast Start Date** are used. If there are none, all deliveries are used instead and the console prints a warning
 - Late rates are worked out for the whole history, for each vendor, and for each Material + Vendor pair. Thin records are pulled toward the broader rate: a pair toward its vendor, and a vendor toward the overall rate. The pull is worth `PRIOR_STRENGTH` (20) deliveries, so a pair with 20 deliveries counts its own record and its vendor's equally, and a pair with 2 is almost entirely its vendor's rate. This stops a material with 3 on-time deliveries from looking 100% reliable
 - **On-Time Probability = 1 - the late rate.** A pair with no history gets its vendor's rate, and an unknown vendor gets the overall rate
 - `On-Time Sample Size` reports how many deliveries were behind the pair's own record (0 means the estimate is entirely the vendor's or the overall rate)
@@ -262,7 +262,17 @@
 Run separately (`python performance_check.py`) after deliveries have happened. File dialogs ask for:
 
 1. The prediction file (`vendor_commit_risk.xlsx`, read from its **Risk Output** sheet)
-2. A receipts file with `Material`, `Document Date`, and `Quantity` columns (a different format from the Historical Commits file)
+2. A receipts file (Excel or CSV), in any layout. This is a different file from Historical Commits: one row per goods receipt
+
+A mapping window then asks which receipts column holds each of the three fields it needs. Each field must use a different column.
+
+| Field | Pre-selected if your file has |
+|---|---|
+| Material | `Material`, `Material Number` or `Material No` |
+| Quantity | `Quantity`, `Qty in Un. of Entry`, `Qty` or `Quantity in Entry Unit` |
+| Document Date | `Document Date`, `Posting Date` or `Date` |
+
+If none of these names are found, the first column is pre-selected, so check every dropdown before confirming. Material is trimmed and upper-cased on both files. **Every receipt quantity is counted as positive** (the absolute value is taken). Reversals and returns that appear as negative quantities would therefore count as extra receipts and make commits look fulfilled, so filter them out of the receipts file first (for example by movement type). The `4.27to6.9` receipts file has no negative quantities.
 
 For each commit row, receipts are counted up to the commit date, with each period starting after the previous commit date for that material. Surplus and shortfall carry forward to the next period. Then:
 
@@ -271,9 +281,9 @@
 - `Actual_Risk = LOW if the ratio is 1.0 or more, otherwise HIGH`
 - `Performance` is CORRECT when the predicted `Risk` matches `Actual_Risk`
 
-A commit filled only after its commit date has a low ratio, so it counts as HIGH. The output also carries `Has_Late_Delivery`, which is TRUE when the receipts file has any receipt for the material dated after the commit date. It is informational: it does not feed `Actual_Risk` or any metric, and can be deleted from the script.
+A commit filled only after its commit date has a low ratio, so it counts as HIGH.
 
-Only commits dated inside the receipts file's date range are scored. **Output:** an Excel file with **Risk Output** (with actuals appended), **Confusion Matrix** (HIGH/LOW), **Metrics** (records, accuracy, HIGH precision, recall, F1), and **Classification Report**.
+Only commits dated inside the receipts file's date range are scored. **Output:** a save dialog asks where to write an Excel file (default name `risk_output_with_actuals.xlsx`) with **Risk Output** (with actuals appended), **Confusion Matrix** (HIGH/LOW), **Metrics** (records, accuracy, HIGH precision, recall, F1), and **Classification Report**.
 
 ### `vendor-commit-eda.py` - Exploratory Analysis (optional)
 A standalone script. It opens a file picker and a column-selection window, then writes a Word (`.docx`) report of tables and control charts. It is not used by the app.
@@ -334,6 +344,8 @@
 | Lateness scorecard | 61.2% | 56.4% | 48.7% |
 | Miss model (current default) | 64.0% | 58.2% | 62.3% |
 
+The quantity-forecast and miss-model rows come from the saved runs `Data/Performance/risk_output_with_actuals.56.xlsx` and `risk_output_with_actuals.55.xlsx`. The lateness-scorecard row, and the month-by-month 2025 test described below, come from earlier analysis that has no saved file in `Data/Performance/`.
+
 For the current default the full report is: HIGH precision 58.2%, recall 62.3%, F1 60.2%; LOW precision 69.1%, recall 65.4%, F1 67.2%. Roughly 47% of commits are flagged HIGH.
 
 **How to read this:** precision is the share of HIGH-flagged commits that turned out to be HIGH. The previous logic's precision (39.6%) was below the share of commits that were actually HIGH (43.6%), so its HIGH flag told you nothing. The current model's (58.2%) is well above it. The previous logic also called almost everything HIGH: it recognised only 23% of the commits that turned out fine, against 65% now. A month-by-month test through 2025 on the historical data, scoring commits that were still open at the start of each month, also had the model ahead of the scorecard in 10 of 11 months.
@@ -343,7 +355,7 @@
 - The 85% and 90% cutoffs were chosen after looking at these same commits, so the figures are probably a little optimistic.
 - The real test is a new batch of commits, scored without changing anything.
 
-Earlier saved runs in `Data/Performance/` used the previous quantity-based logic.
+Saved runs in `Data/Performance/`: `risk_output_with_actuals.55.xlsx` is the miss model (current default). The others, including `.56`, use the previous quantity-based logic. You can tell them apart by the columns: only miss-model runs have `Model On-Time Probability` and `On-Time Probability`.
 
 ---
 
@@ -373,6 +385,7 @@
 - Shortfalls count as misses, but they are rare in the history (297 of 96,586 rows in the 12/31/25 extract), so the model is mostly learning lateness.
 - `performance_check.py` scores a commit by cumulative quantity received by the commit date, with carry-over between commits. The model predicts a single commit's outcome, so the two don't always line up. In the test set about a quarter of the commits that were actually HIGH were partial receipts rather than fully late.
 - The risk cutoffs were tuned on one small test set (see [Model Performance](#model-performance)).
+- `performance_check.py` counts every receipt quantity as positive, so negative rows (reversals, returns) in the receipts file inflate fulfilment. Remove them before scoring.
 - A pair with no history gets its vendor's rate, and an unknown vendor the overall rate (`On-Time Sample Size` is 0 for these).
 - For materials with more than one vendor, the lateness scorecard and the miss model use the latest vendor seen for the material.
 - The miss model is retrained on every run and is not saved.
@@ -382,7 +395,7 @@
 - The delivery description score is the lateness of the same delivery whose quantity is being predicted, not earlier history, and it is replaced by the average at forecast time.
 - `sigma` is measured on the training data, so it underestimates true forecast error, and it is the same for every day of the window. A walk-forward (out-of-sample) sigma is planned.
 - Tree models cannot extrapolate the date trend beyond the training range.
-- A Material + Vendor pair needs at least 5 historical rows; otherwise it is skipped with no message, and its commits are dropped from the output.
+- A Material + Vendor pair needs at least 5 historical rows; otherwise it is skipped (a console line only, nothing in the browser), and its commits are dropped from the output.
 - It still trains on every historical row, including any dated after the Forecast Start Date.
 - The quantity probability scores missing inputs as 0% / LOW, and `sigma = 0` as 50% (HIGH). The confidence cutoffs are fixed constants.
 - The accuracy metrics ignore over-prediction and are computed once per run.
