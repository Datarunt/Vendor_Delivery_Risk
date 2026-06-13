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
cum_actuals = []
cum_commits = []
late_flags = []

for idx, row in risk_df.iterrows():

    material = row["Material"]
    vendor_commit = row["Vendor Commit"]

    # historical data for material
    hist_material = hist_df[
        hist_df["Material"] == material
    ]

    # risk rows for material
    risk_material = risk_df[
        (risk_df["Material"] == material) &
        (risk_df["Vendor Commit"] <= vendor_commit)
    ]

    # cumulative actuals up to commit date
    actual_total = hist_material[
        hist_material["Document Date"] <= vendor_commit
    ]["Quantity"].sum()

    # cumulative commits up to commit date
    commit_total = risk_material["Commit Qty"].sum()

    # late deliveries
    has_late = (
        hist_material["Document Date"] > vendor_commit
    ).any()

    cum_actuals.append(actual_total)
    cum_commits.append(commit_total)
    late_flags.append(has_late)

# -----------------------------
# ADD METRICS
# -----------------------------
risk_df["Cum_Actual"] = cum_actuals
risk_df["Cum_Commit"] = cum_commits
risk_df["Has_Late_Delivery"] = late_flags

# -----------------------------
# PERFORMANCE LOGIC
# -----------------------------
risk_df["Performance"] = "INCORRECT"

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
risk_df["Actual_Risk"] = np.select(
    [
        risk_df["Fulfillment_Ratio"] < 0.80,

        (risk_df["Fulfillment_Ratio"] >= 0.80) &
        (risk_df["Fulfillment_Ratio"] < 1.00),

        risk_df["Fulfillment_Ratio"] >= 1.00
    ],
    [
        "HIGH",
        "MED",
        "LOW"
    ],
    default="UNKNOWN"
)

# -----------------------------
# MED correct if ratio between
# 80% and 99%
# -----------------------------
risk_df.loc[
    (risk_df["Risk"] == "MED") &
    (
        (risk_df["Fulfillment_Ratio"] >= 0.80) &
        (risk_df["Fulfillment_Ratio"] < 1.00)
    ),
    "Performance"
] = "CORRECT"

# LOW correct
risk_df.loc[
    (risk_df["Risk"] == "LOW") &
    (risk_df["Cum_Actual"] >= risk_df["Cum_Commit"]),
    "Performance"
] = "CORRECT"

# HIGH correct
risk_df.loc[
    (risk_df["Risk"] == "HIGH") &
    (risk_df["Cum_Actual"] < risk_df["Cum_Commit"]),
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
labels = ["HIGH", "MED", "LOW"]

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
    labels=["HIGH", "MED", "LOW"],
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