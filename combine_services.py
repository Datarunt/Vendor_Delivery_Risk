import pandas as pd
from risk_engine import apply_risk_model

def build_combined_output(forecast_df,commits_df,owner_df,forecast_start_date,forecast_horizon):

    # DEBUG
    print("Commits columns:", commits_df.columns.tolist())
    print("Commits sample M00012:")
    print(commits_df[commits_df["Material"].astype(str).str.strip().str.upper() == "M00012"][["Material","Vendor","Vendor Name"]].head() if "Vendor Name" in commits_df.columns else "NO VENDOR NAME COLUMN")

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
    agg_dict = {"Commit Qty": "sum"}
    if "Vendor" in commits_window.columns:
        agg_dict["Vendor"] = "first"
    if "Vendor Name" in commits_window.columns:
        agg_dict["Vendor Name"] = "first"

    commits_agg = commits_window.groupby(
        ["Material", "Commit Dt by Suppl"],
        as_index=False
    ).agg(agg_dict)

    # ---------------------------
    merged = commits_agg.merge(
        forecast_window,
        left_on=["Material", "Commit Dt by Suppl"],
        right_on=["Material", "Date Received"],
        how="left"
    )

    if "Vendor_x" in merged.columns:
        merged["Vendor"] = merged["Vendor_x"]

    # resolve Vendor Name conflict from merge
    if "Vendor Name_x" in merged.columns:
        merged["Vendor Name"] = merged["Vendor Name_x"].fillna(
            merged.get("Vendor Name_y", pd.Series(dtype=str))
        )
    elif "Vendor Name" not in merged.columns:
        merged["Vendor Name"] = None

    # ---------------------------
    # OWNER MATRIX JOIN
    def norm(x):
        return str(x).strip().replace(".0", "").lstrip("0")

    merged["Vendor"] = merged["Vendor"].apply(norm)

    owner_df.columns = owner_df.columns.str.strip()
    owner_df["Supplier #"] = owner_df["Supplier #"].apply(norm)

    supplier_to_name = dict(
        zip(owner_df["Supplier #"], owner_df.get("Supplier Name", owner_df.get("Vendor Name", pd.Series(dtype=str))))
    )
    supplier_to_buyer = dict(
        zip(owner_df["Supplier #"], owner_df["Assigned Buyer"])
    )

    # fill Vendor Name — prefer commits, then owner matrix, then vendor ID
    merged["Vendor Name"] = merged["Vendor Name"].fillna(
        merged["Vendor"].map(supplier_to_name)
    ).fillna(merged["Vendor"])

    merged["Assigned Buyer"] = merged["Vendor"].map(supplier_to_buyer)
    merged["Assigned Buyer"] = merged["Assigned Buyer"].fillna("UNKNOWN")

    merged = merged[
        merged["Material"].notna() &
        merged["Quantity Received"].notna()
    ]

    merged = apply_risk_model(merged)

    # ---------------------------
    # Final Output
    merged["Commit Dt by Suppl"] = pd.to_datetime(merged["Commit Dt by Suppl"]).dt.strftime("%Y-%m-%d")
    final = merged.rename(columns={"Commit Dt by Suppl": "Vendor Commit"})
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