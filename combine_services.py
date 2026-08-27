import pandas as pd
import numpy as np
from risk_engine import apply_risk_model

# ---------------------------
# COMMIT DATE AGGREGATION
def aggregate_commit_dates(dates, aggregation):
    """
    Bucket dates according to aggregation level.

    day   -> original date
    week  -> coming Sunday
    month -> month end
    """
    aggregation = (aggregation or "day").strip().lower()

    if aggregation == "week":
        days_until_sunday = (6 - dates.dt.dayofweek) % 7
        return dates + pd.to_timedelta(days_until_sunday, unit="D")

    elif aggregation == "month":
        return dates + pd.offsets.MonthEnd(0)

    return dates


# ---------------------------
# MAIN FUNCTION
def build_combined_output(
    forecast_df,
    commits_df,
    owner_df,
    forecast_start_date,
    forecast_horizon,
    aggregation="day"
):

    aggregation = (aggregation or "day").strip().lower()

    # ---------------------------
    # Work on copies so we don't modify source DataFrames
    forecast_df = forecast_df.copy()
    commits_df = commits_df.copy()
    owner_df = owner_df.copy()

    # ---------------------------
    # Clean Data
    forecast_df["Material"] = (
        forecast_df["Material"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    commits_df["Material"] = (
        commits_df["Material"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    forecast_df["Date Received"] = pd.to_datetime(
        forecast_df["Date Received"],
        errors="coerce"
    )

    commits_df["Commit Dt by Suppl"] = pd.to_datetime(
        commits_df["Commit Dt by Suppl"],
        errors="coerce"
    )

    forecast_df["Quantity Received"] = pd.to_numeric(
        forecast_df["Quantity Received"],
        errors="coerce"
    )

    if "Sigma" in forecast_df.columns:
        forecast_df["Sigma"] = pd.to_numeric(
            forecast_df["Sigma"],
            errors="coerce"
        )

    commits_df["Commit Qty"] = pd.to_numeric(
        commits_df["Commit Qty"],
        errors="coerce"
    )

    # Normalize forecast start date
    start = pd.to_datetime(forecast_start_date)
    end = start + pd.Timedelta(days=forecast_horizon - 1)

    # ============================================================
    # COMMITS
    # ============================================================

    # Filter using ORIGINAL commit date BEFORE aggregation.
    commits_window = commits_df[
        (commits_df["Commit Dt by Suppl"] >= start) &
        (commits_df["Commit Dt by Suppl"] <= end)
    ].copy()

    # Bucket commit dates
    commits_window["Commit Dt by Suppl"] = aggregate_commit_dates(
        commits_window["Commit Dt by Suppl"],
        aggregation
    )

    # Aggregate commits
    commit_agg_dict = {
        "Commit Qty": "sum"
    }

    if "Vendor" in commits_window.columns:
        commit_agg_dict["Vendor"] = "first"

    if "Vendor Name" in commits_window.columns:
        commit_agg_dict["Vendor Name"] = "first"

    commits_agg = commits_window.groupby(
        ["Material", "Commit Dt by Suppl"],
        as_index=False
    ).agg(commit_agg_dict)

    # ============================================================
    # FORECAST
    # ============================================================

    forecast_window = forecast_df[
        (forecast_df["Date Received"] >= start) &
        (forecast_df["Date Received"] <= end) &
        (forecast_df["is_forecast"] == True)
    ].copy()

    # ------------------------------------------------------------
    # Deduplicate daily forecasts.
    #
    # Keep the forecast with the lowest Sigma for each
    # Material + actual date.
    # ------------------------------------------------------------
    if "Sigma" in forecast_window.columns:

        forecast_window = (
            forecast_window
            .sort_values(
                ["Material", "Date Received", "Sigma"],
                na_position="last"
            )
            .drop_duplicates(
                subset=["Material", "Date Received"],
                keep="first"
            )
        )

    else:
        forecast_window = forecast_window.drop_duplicates(
            subset=["Material", "Date Received"],
            keep="first"
        )

    # ------------------------------------------------------------
    # Bucket forecast dates using EXACT SAME function as commits.
    #
    # Keep the original (pre-bucket) date around as "_orig_date" so
    # that, within each bucket, we can identify which day's forecast
    # sits closest to the bucket's own reference date (the Sunday /
    # month-end that the commit side is also bucketed to).
    # ------------------------------------------------------------
    forecast_window["_orig_date"] = forecast_window["Date Received"]

    forecast_window["Date Received"] = aggregate_commit_dates(
        forecast_window["Date Received"],
        aggregation
    )

    # ------------------------------------------------------------
    # Pick ONE representative forecast per Material + bucket,
    # instead of summing every daily forecast in the bucket.
    #
    # WHY: the underlying model (model.py) predicts a full delivery-
    # sized quantity for EVERY calendar day, since it was trained
    # only on days where a delivery actually happened and has no
    # concept of "no delivery expected today." At Day aggregation
    # that's fine -- one day's prediction is a reasonable stand-in
    # for "a delivery around here." But summing 7 (week) or ~30
    # (month) of those daily predictions together inflated the
    # forecast by roughly the number of days in the bucket, which is
    # what was producing mu values 7x-30x larger than Commit Qty and
    # pinning Probability at 100% for nearly every row.
    #
    # Fix: within each Material + bucket, keep only the forecast
    # from the day closest to the bucket's reference date (the same
    # Sunday / month-end the commit is bucketed to). Since every
    # date within a bucket is <= that reference date, "closest" is
    # just the latest (most recent) day in the bucket.
    #
    # At Day aggregation this is a no-op: each bucket already
    # contains exactly one row (thanks to the per-day dedup above),
    # so "pick the representative row" and "sum" give identical
    # results -- Day output is unaffected by this change.
    # ------------------------------------------------------------
    forecast_window = (
        forecast_window
        .sort_values(["Material", "Date Received", "_orig_date"])
        .drop_duplicates(
            subset=["Material", "Date Received"],
            keep="last"
        )
        .drop(columns=["_orig_date"])
    )

    # ------------------------------------------------------------
    # Keep only the columns needed downstream. No aggregation
    # math is required here anymore -- Quantity Received and Sigma
    # are simply the representative single day's own values.
    # ------------------------------------------------------------
    forecast_cols = ["Material", "Date Received", "Quantity Received"]

    if "Sigma" in forecast_window.columns:
        forecast_cols.append("Sigma")

    if "Vendor Name" in forecast_window.columns:
        forecast_cols.append("Vendor Name")

    forecast_window = forecast_window[forecast_cols]

    # ============================================================
    # MERGE
    # ============================================================

    merged = commits_agg.merge(
        forecast_window,
        left_on=["Material", "Commit Dt by Suppl"],
        right_on=["Material", "Date Received"],
        how="left",
        indicator=True
    )

    # ------------------------------------------------------------
    # Check for unmatched forecast buckets.
    # ------------------------------------------------------------
    unmatched = merged[merged["_merge"] == "left_only"]

    if not unmatched.empty:

        print(
            f"WARNING: {len(unmatched)} commit bucket(s) "
            f"have no matching forecast bucket."
        )

        print(
            unmatched[
                [
                    "Material",
                    "Commit Dt by Suppl",
                    "Commit Qty"
                ]
            ].head(20).to_string(index=False)
        )

    merged = merged.drop(columns=["_merge"])

    # ============================================================
    # VENDOR
    # ============================================================

    if "Vendor_x" in merged.columns:
        merged["Vendor"] = merged["Vendor_x"]

    elif "Vendor" not in merged.columns:
        merged["Vendor"] = None

    # ------------------------------------------------------------
    # Vendor Name
    # ------------------------------------------------------------

    if "Vendor Name_x" in merged.columns:

        if "Vendor Name_y" in merged.columns:
            merged["Vendor Name"] = (
                merged["Vendor Name_x"]
                .fillna(merged["Vendor Name_y"])
            )
        else:
            merged["Vendor Name"] = merged["Vendor Name_x"]

    elif "Vendor Name" not in merged.columns:

        if "Vendor Name_y" in merged.columns:
            merged["Vendor Name"] = merged["Vendor Name_y"]
        else:
            merged["Vendor Name"] = None

    # ============================================================
    # OWNER MATRIX
    # ============================================================

    def norm(x):
        if pd.isna(x):
            return ""

        return (
            str(x)
            .strip()
            .replace(".0", "")
            .lstrip("0")
        )

    merged["Vendor"] = merged["Vendor"].apply(norm)

    owner_df.columns = owner_df.columns.str.strip()

    owner_df["Supplier #"] = owner_df["Supplier #"].apply(norm)

    # ------------------------------------------------------------
    # Supplier -> Name
    # ------------------------------------------------------------

    if "Supplier Name" in owner_df.columns:
        supplier_names = owner_df["Supplier Name"]

    elif "Vendor Name" in owner_df.columns:
        supplier_names = owner_df["Vendor Name"]

    else:
        supplier_names = pd.Series(
            index=owner_df.index,
            dtype=object
        )

    supplier_to_name = dict(
        zip(
            owner_df["Supplier #"],
            supplier_names
        )
    )

    # ------------------------------------------------------------
    # Supplier -> Buyer
    # ------------------------------------------------------------

    supplier_to_buyer = dict(
        zip(
            owner_df["Supplier #"],
            owner_df["Assigned Buyer"]
        )
    )

    # ------------------------------------------------------------
    # Fill Vendor Name
    #
    # Priority:
    # 1. Commit Vendor Name
    # 2. Owner Matrix
    # 3. Vendor ID
    # ------------------------------------------------------------

    merged["Vendor Name"] = (
        merged["Vendor Name"]
        .fillna(merged["Vendor"].map(supplier_to_name))
        .fillna(merged["Vendor"])
    )

    # ------------------------------------------------------------
    # Assigned Buyer
    # ------------------------------------------------------------

    merged["Assigned Buyer"] = (
        merged["Vendor"]
        .map(supplier_to_buyer)
        .fillna("UNKNOWN")
    )

    # ============================================================
    # VALID RISK MODEL INPUT
    # ============================================================

    # Material must exist
    merged = merged[
        merged["Material"].notna()
    ].copy()

    # Quantity Received must exist
    merged = merged[
        merged["Quantity Received"].notna()
    ].copy()

    # Commit Qty should be numeric
    merged["Commit Qty"] = pd.to_numeric(
        merged["Commit Qty"],
        errors="coerce"
    )

    # ============================================================
    # DEBUG INFORMATION
    # ============================================================

    print(
        f"\nRisk model input - aggregation: {aggregation}"
    )

    print(
        f"Rows: {len(merged)}"
    )

    print(
        "Missing values:"
    )

    print(
        merged[
            [
                "Commit Qty",
                "Quantity Received",
                "Sigma"
            ]
        ].isna().sum()
    )

    # ============================================================
    # APPLY RISK MODEL
    # ============================================================

    merged = apply_risk_model(merged)

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    merged["Commit Dt by Suppl"] = pd.to_datetime(
        merged["Commit Dt by Suppl"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    final = merged.rename(
        columns={
            "Commit Dt by Suppl": "Vendor Commit"
        }
    )

    final = final[
        [
            "Material",
            "Vendor Name",
            "Vendor Commit",
            "Commit Qty",
            "Probability",
            "Risk",
            "Confidence Interval",
            "Confidence",
            "Assigned Buyer"
        ]
    ]

    return final