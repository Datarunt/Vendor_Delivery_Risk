import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# -----------------------------
# FILE PICKERS
# -----------------------------
root = tk.Tk()
root.withdraw()

print("Select prediction file...")
prediction_path = filedialog.askopenfilename(
    title="Select Prediction File",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

print("Select historical receipt file...")
historical_path = filedialog.askopenfilename(
    title="Select Historical Receipt File",
    filetypes=[("Excel files", "*.xlsx *.xls *.csv")]
)

# -----------------------------
# LOAD FILES
# -----------------------------
risk_df = pd.read_excel(
    prediction_path,
    sheet_name="Risk Output"
)

if historical_path.lower().endswith(".csv"):
    hist_df = pd.read_csv(historical_path)
else:
    hist_df = pd.read_excel(historical_path)

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
risk_df.columns = risk_df.columns.str.strip()
hist_df.columns = hist_df.columns.str.strip()

# -----------------------------
# REQUIRED COLUMNS
# -----------------------------
required_cols = [
    "Material",
    "Document Date",
    "Quantity"
]

for col in required_cols:
    if col not in hist_df.columns:
        raise ValueError(f"Missing required column: {col}")

# -----------------------------
# CLEAN DATA
# -----------------------------
risk_df["Material"] = (
    risk_df["Material"]
    .astype(str)
    .str.strip()
    .str.upper()
)

hist_df["Material"] = (
    hist_df["Material"]
    .astype(str)
    .str.strip()
    .str.upper()
)

risk_df["Vendor Commit"] = pd.to_datetime(
    risk_df["Vendor Commit"],
    errors="coerce"
)

hist_df["Document Date"] = pd.to_datetime(
    hist_df["Document Date"],
    errors="coerce"
)

risk_df["Commit Qty"] = pd.to_numeric(
    risk_df["Commit Qty"],
    errors="coerce"
).fillna(0)

hist_df["Quantity"] = pd.to_numeric(
    hist_df["Quantity"],
    errors="coerce"
).fillna(0)

risk_df["Risk"] = (
    risk_df["Risk"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# -----------------------------
# SORT DATA
# -----------------------------
risk_df = risk_df.sort_values(
    by=["Material", "Vendor Commit"]
)

hist_df = hist_df.sort_values(
    by=["Material", "Document Date"]
)

# -----------------------------
# BUILD METRICS
# -----------------------------
# For each material, commit rows are treated as consecutive, non-
# overlapping periods. The first period per material has no lower
# bound -- it's simply every historical receipt up through that first
# commit date. Every later period's OWN actuals are bounded below by
# the previous commit date (exclusive), so a physical receipt is
# never counted as satisfying two different periods at once.
#
# On top of that, a running carryover applies symmetrically:
#   - SURPLUS (a period received more than it needed) rolls forward
#     and adds to the next period's available actuals -- extra units
#     don't just vanish because they arrived "early".
#   - SHORTFALL (a period received less than it needed) rolls forward
#     and adds to the next period's required commit -- a vendor who
#     falls behind stays on the hook for the make-up quantity, it
#     doesn't just reset each period.
risk_df = risk_df.reset_index(drop=True)

cum_actual_map = {}
cum_commit_map = {}
late_flag_map = {}

for material, group in risk_df.groupby("Material", sort=False):

    group = group.sort_values("Vendor Commit")
    hist_material = hist_df[hist_df["Material"] == material]

    previous_commit_date = None
    carry_actual = 0.0   # unused surplus rolling into the next period
    carry_commit = 0.0   # unmet shortfall rolling into the next period

    for idx, row in group.iterrows():

        vendor_commit = row["Vendor Commit"]

        # commit quantity for THIS specific commit date only. Commit
        # data is a point-in-time snapshot (e.g. pulled 4/27) -- each
        # distinct Commit Dt by Suppl is its own separate obligation,
        # not additional demand stacked on top of an earlier-dated
        # commit. Rows sharing the exact same commit date ARE
        # combined (e.g. two same-day split-shipment lines), but
        # different dates are independent and are not summed across
        # each other.
        own_commit = risk_df[
            (risk_df["Material"] == material) &
            (risk_df["Vendor Commit"] == vendor_commit)
        ]["Commit Qty"].sum()

        # this period's own actuals only
        if previous_commit_date is None:
            period_mask = (
                hist_material["Document Date"] <= vendor_commit
            )
        else:
            period_mask = (
                (hist_material["Document Date"] > previous_commit_date) &
                (hist_material["Document Date"] <= vendor_commit)
            )

        own_actual = hist_material[period_mask]["Quantity"].sum()

        # apply carry-in from the previous period
        commit_total = own_commit + carry_commit
        actual_total = own_actual + carry_actual

        has_late = (
            hist_material["Document Date"] > vendor_commit
        ).any()

        cum_actual_map[idx] = actual_total
        cum_commit_map[idx] = commit_total
        late_flag_map[idx] = has_late

        # compute what carries into the NEXT period
        carry_actual = max(actual_total - commit_total, 0.0)
        carry_commit = max(commit_total - actual_total, 0.0)

        previous_commit_date = vendor_commit

# -----------------------------
# ADD METRICS
# -----------------------------
risk_df["Cum_Actual"] = risk_df.index.map(cum_actual_map)
risk_df["Cum_Commit"] = risk_df.index.map(cum_commit_map)
risk_df["Has_Late_Delivery"] = risk_df.index.map(late_flag_map)

# -----------------------------
# FULFILLMENT RATIO
# -----------------------------
risk_df["Fulfillment_Ratio"] = (
    risk_df["Cum_Actual"] /
    risk_df["Cum_Commit"]
).fillna(0)

# -----------------------------
# ACTUAL RISK CLASSIFICATION
# -----------------------------
# NOTE: risk_engine.py collapsed HIGH/MED/LOW down to a binary
# HIGH/LOW scheme (probability of meeting commitment < 75% = HIGH,
# otherwise LOW). The ground-truth label here is rebuilt to match
# that same 2-class scheme -- "did the vendor actually meet or
# exceed what they committed to, yes or no" -- instead of the old
# 3-tier version, which could never be scored as correct once the
# model stopped predicting MED at all.
risk_df["Actual_Risk"] = np.where(
    risk_df["Fulfillment_Ratio"] >= 1.00,
    "LOW",
    "HIGH"
)

# -----------------------------
# PERFORMANCE LOGIC
# -----------------------------
risk_df["Performance"] = "INCORRECT"

# LOW correct
risk_df.loc[
    (risk_df["Risk"] == "LOW") &
    (risk_df["Actual_Risk"] == "LOW"),
    "Performance"
] = "CORRECT"

# HIGH correct
risk_df.loc[
    (risk_df["Risk"] == "HIGH") &
    (risk_df["Actual_Risk"] == "HIGH"),
    "Performance"
] = "CORRECT"

# -----------------------------
# FINAL OUTPUT
# -----------------------------
final_df = risk_df.copy()

# -----------------------------
# FILTER FINAL OUTPUT TO
# HISTORICAL DOCUMENT DATE RANGE
# -----------------------------
min_doc_date = hist_df["Document Date"].min()
max_doc_date = hist_df["Document Date"].max()

final_df = final_df[
    (final_df["Vendor Commit"] >= min_doc_date) &
    (final_df["Vendor Commit"] <= max_doc_date)
].copy()

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
labels = ["HIGH", "LOW"]

cm = confusion_matrix(
    final_df["Actual_Risk"],
    final_df["Risk"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Predicted_{x}" for x in labels]
)

# -----------------------------
# PRECISION / RECALL / F1
# -----------------------------
report = classification_report(
    final_df["Actual_Risk"],
    final_df["Risk"],
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

high_precision = report["HIGH"]["precision"]
high_recall = report["HIGH"]["recall"]
high_f1 = report["HIGH"]["f1-score"]

# -----------------------------
# CLASSIFICATION ACCURACY
# -----------------------------
classification_accuracy = accuracy_score(
    final_df["Actual_Risk"],
    final_df["Risk"]
)

summary_df = pd.DataFrame({
    "Metric": [
        "Records Evaluated",
        "Classification Accuracy",
        "High Risk Precision",
        "High Risk Recall",
        "High Risk F1"
    ],
    "Value": [
        len(final_df),
        f"{classification_accuracy:.2%}",
        f"{high_precision:.2%}",
        f"{high_recall:.2%}",
        f"{high_f1:.2%}"
    ]
})

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_path = filedialog.asksaveasfilename(
    title="Save Output File",
    defaultextension=".xlsx",
    filetypes=[("Excel files", "*.xlsx")],
    initialfile="risk_output_with_actuals.xlsx"
)

if not output_path:
    raise SystemExit("Save cancelled.")

with pd.ExcelWriter(
    output_path,
    engine="xlsxwriter"
) as writer:

    final_df.to_excel(
    writer,
    sheet_name="Risk Output",
    index=False
)

    cm_df.to_excel(
        writer,
        sheet_name="Confusion Matrix"
)
    
    summary_df.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
)
    
    report_df.to_excel(
        writer,
        sheet_name="Classification Report"
)

print("\nDONE")
print("Saved file:")
print(output_path)