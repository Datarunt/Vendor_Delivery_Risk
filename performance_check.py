import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
 
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
 
if not prediction_path:
    root.destroy()
    raise SystemExit("Prediction file selection cancelled.")
 
print("Select historical receipt file...")
historical_path = filedialog.askopenfilename(
    title="Select Historical Receipt File",
    filetypes=[
        ("Excel and CSV files", "*.xlsx *.xls *.csv")
    ]
)
 
if not historical_path:
    root.destroy()
    raise SystemExit("Historical file selection cancelled.")
 
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
risk_df.columns = risk_df.columns.astype(str).str.strip()
hist_df.columns = hist_df.columns.astype(str).str.strip()
 
# -----------------------------
# COLUMN MAPPING WINDOW
# -----------------------------
# Select the historical columns that correspond to:
# Material, Quantity, and Document Date.
 
column_names = list(hist_df.columns)
 
# Attempt to identify sensible defaults automatically.
def find_default(candidates, columns, fallback_index=0):
    lookup = {str(col).strip().lower(): col for col in columns}
 
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
 
    return columns[fallback_index] if columns else ""
 
material_default = find_default(
    ["Material", "Material Number", "Material No"],
    column_names
)
 
quantity_default = find_default(
    ["Quantity", "Qty in Un. of Entry", "Qty", "Quantity in Entry Unit"],
    column_names
)
 
date_default = find_default(
    ["Document Date", "Posting Date", "Posting date", "Date"],
    column_names
)
 
# Create a visible mapping window.
root.deiconify()
root.title("Historical Receipt Column Mapping")
root.geometry("560x300")
root.resizable(False, False)
 
tk.Label(
    root,
    text="Map Historical Receipt Columns",
    font=("Arial", 14, "bold")
).pack(pady=(15, 5))
 
tk.Label(
    root,
    text=(
        "Choose the columns in your historical dataset that match\n"
        "each required field."
    ),
    justify="center"
).pack(pady=(0, 12))
 
mapping_frame = tk.Frame(root)
mapping_frame.pack(padx=25, fill="x")
 
material_var = tk.StringVar(value=material_default)
quantity_var = tk.StringVar(value=quantity_default)
date_var = tk.StringVar(value=date_default)
 
def add_dropdown(label_text, variable):
    row = tk.Frame(mapping_frame)
    row.pack(fill="x", pady=6)
 
    tk.Label(
        row,
        text=label_text,
        width=18,
        anchor="w"
    ).pack(side="left")
 
    dropdown = ttk.Combobox(
        row,
        textvariable=variable,
        values=column_names,
        state="readonly",
        width=38
    )
    dropdown.pack(side="left", fill="x", expand=True)
 
add_dropdown("Material:", material_var)
add_dropdown("Quantity:", quantity_var)
add_dropdown("Document Date:", date_var)
 
def confirm_mapping():
    selected = [
        material_var.get(),
        quantity_var.get(),
        date_var.get()
    ]
 
    if not all(selected):
        messagebox.showerror(
            "Missing Selection",
            "Please select a column for all three fields.",
            parent=root
        )
        return
 
    if len(set(selected)) != 3:
        messagebox.showerror(
            "Duplicate Selection",
            "Each field must map to a different column.",
            parent=root
        )
        return
 
    root.quit()
 
tk.Button(
    root,
    text="Confirm Mapping",
    command=confirm_mapping,
    width=20
).pack(pady=18)
 
root.mainloop()
 
# Capture the selections before closing the window.
material_col = material_var.get()
quantity_col = quantity_var.get()
date_col = date_var.get()
 
root.destroy()
 
# -----------------------------
# STANDARDIZE HISTORICAL COLUMNS
# -----------------------------
hist_df = hist_df.rename(columns={
    material_col: "Material",
    quantity_col: "Quantity",
    date_col: "Document Date"
})
 
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
    .astype("string")
    .str.strip()
    .str.upper()
)
 
hist_df["Material"] = (
    hist_df["Material"]
    .astype("string")
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
 
# Convert historical quantities to numeric, then absolute values.
hist_df["Quantity"] = pd.to_numeric(
    hist_df["Quantity"],
    errors="coerce"
).abs().fillna(0)
 
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
risk_df = risk_df.reset_index(drop=True)
 
cum_actual_map = {}
cum_commit_map = {}
 
for material, group in risk_df.groupby("Material", sort=False):
 
    group = group.sort_values("Vendor Commit")
    hist_material = hist_df[
        hist_df["Material"] == material
    ]
 
    previous_commit_date = None
    carry_actual = 0.0
    carry_commit = 0.0
 
    for idx, row in group.iterrows():
 
        vendor_commit = row["Vendor Commit"]
 
        # Sum commit quantities only for this material and date.
        own_commit = risk_df[
            (risk_df["Material"] == material) &
            (risk_df["Vendor Commit"] == vendor_commit)
        ]["Commit Qty"].sum()
 
        # Actual receipts for this period only.
        if previous_commit_date is None:
            period_mask = (
                hist_material["Document Date"] <= vendor_commit
            )
        else:
            period_mask = (
                (hist_material["Document Date"] > previous_commit_date) &
                (hist_material["Document Date"] <= vendor_commit)
            )
 
        own_actual = hist_material.loc[
            period_mask, "Quantity"
        ].sum()
 
        # Apply carryover from previous periods.
        commit_total = own_commit + carry_commit
        actual_total = own_actual + carry_actual
 
        cum_actual_map[idx] = actual_total
        cum_commit_map[idx] = commit_total
 
        # Carry surplus or shortfall forward.
        carry_actual = max(actual_total - commit_total, 0.0)
        carry_commit = max(commit_total - actual_total, 0.0)
 
        previous_commit_date = vendor_commit
 
# -----------------------------
# ADD METRICS
# -----------------------------
risk_df["Cum_Actual"] = risk_df.index.map(cum_actual_map)
risk_df["Cum_Commit"] = risk_df.index.map(cum_commit_map)
 
# -----------------------------
# FULFILLMENT RATIO
# -----------------------------
risk_df["Fulfillment_Ratio"] = (
    risk_df["Cum_Actual"] /
    risk_df["Cum_Commit"].replace(0, np.nan)
).fillna(0)
 
# -----------------------------
# ACTUAL RISK CLASSIFICATION
# -----------------------------
risk_df["Actual_Risk"] = np.where(
    risk_df["Fulfillment_Ratio"] >= 1.00,
    "LOW",
    "HIGH"
)
 
# -----------------------------
# PERFORMANCE LOGIC
# -----------------------------
risk_df["Performance"] = np.where(
    risk_df["Risk"] == risk_df["Actual_Risk"],
    "CORRECT",
    "INCORRECT"
)
 
# -----------------------------
# FINAL OUTPUT
# -----------------------------
final_df = risk_df.copy()
 
# Filter to historical document date range.
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
root = tk.Tk()
root.withdraw()
 
output_path = filedialog.asksaveasfilename(
    title="Save Output File",
    defaultextension=".xlsx",
    filetypes=[("Excel files", "*.xlsx")],
    initialfile="risk_output_with_actuals.xlsx"
)
 
root.destroy()
 
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

