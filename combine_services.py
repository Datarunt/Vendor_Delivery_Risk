import pandas as pd
from risk_engine import apply_risk_model

def build_combined_output(forecast_df,commits_df,owner_df,forecast_start_date,forecast_horizon):

    # ---------------------------
    # Clean Data
    forecast_df["Material"] = forecast_df["Material"].astype(str).str.strip().str.upper()
    commits_df["Material"] = commits_df["Material"].astype(str).str.strip().str.upper()
    forecast_df["Date Received"] = pd.to_datetime(forecast_df["Date Received"])
    commits_df["Commit Dt by Suppl"] = pd.to_datetime(commits_df["Commit Dt by Suppl"], errors="coerce")
    commits_df["Commit Qty"] = pd.to_numeric(commits_df["Commit Qty"], errors="coerce")

    start = forecast_start_date
    end = start + pd.Timedelta(days=forecast_horizon - 1)

    commits_window = commits_df[(commits_df["Commit Dt by Suppl"] >= start) &
                                (commits_df["Commit Dt by Suppl"] <= end)].copy()
    forecast_window = forecast_df[(forecast_df["Date Received"] >= start) &
                                  (forecast_df["Date Received"] <= end) &
                                  (forecast_df["is_forecast"] == True)].copy()

    # dedup - keep row with lowest sigma (most confident forecast)
    forecast_window = forecast_window.sort_values("Sigma").drop_duplicates(
        subset=["Material", "Date Received"], keep="first"
    )

    # ---------------------------
    # Aggregate Commits First
    commits_agg = commits_window.groupby(
        ["Material", "Commit Dt by Suppl"],
        as_index=False
    ).agg({
        "Commit Qty": "sum",
        "Vendor": "first" if "Vendor" in commits_window.columns else None
    })

    # ---------------------------
    merged = commits_agg.merge(
        forecast_window,
        left_on=["Material", "Commit Dt by Suppl"],
        right_on=["Material", "Date Received"],
        how="left"
    )

    if "Vendor_x" in merged.columns:
        merged["Vendor"] = merged["Vendor_x"]

    if "Vendor Name" not in merged.columns:
        merged["Vendor Name"] = merged["Vendor"]
    else:
        merged["Vendor Name"] = merged["Vendor Name"].fillna(merged["Vendor"])

    # ---------------------------
    # OWNER MATRIX JOIN
    def norm(x):
        return str(x).strip().replace(".0", "").lstrip("0")

    merged["Vendor"] = merged["Vendor"].apply(norm)

    owner_df.columns = owner_df.columns.str.strip()
    owner_df["Supplier #"] = owner_df["Supplier #"].apply(norm)

    supplier_to_buyer = dict(
        zip(owner_df["Supplier #"], owner_df["Assigned Buyer"])
    )

    merged["Assigned Buyer"] = merged["Vendor"].map(supplier_to_buyer)
    merged["Assigned Buyer"] = merged["Assigned Buyer"].fillna("UNKNOWN")

    merged = merged[
        merged["Material"].notna() &
        merged["Quantity Received"].notna()
    ]

    merged = apply_risk_model(merged)

    # ---------------------------
    # Final Output
    final = merged.rename(columns={"Commit Dt by Suppl": "Vendor Commit"})
    final["Vendor Commit"] = pd.to_datetime(final["Vendor Commit"]).dt.date
    final = final[[
        "Material",
        "Vendor Name",
        "Vendor Commit",
        "Commit Qty",
        "Probability",
        "Risk",
        "Confidence Interval",
        "Confidence",
        "Assigned Buyer"
    ]]

    return final