import pandas as pd
import numpy as np
from risk_engine import apply_risk_model
import risk_engine
from lateness_services import build_lateness_table, add_on_time_probability
from ml_risk_services import choose_lead_days, train_miss_model, predict_on_time_probability

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
    # HISTORICAL LATENESS
    #
    # The history rows (is_forecast == False) carry each delivery's
    # Working Days Late. Only deliveries BEFORE the forecast start
    # are used, so a commit is never judged with deliveries that
    # happened after it.
    # ============================================================

    lateness_table = None
    miss_model = None

    if "Working Days Late" in forecast_df.columns:
        history_rows = forecast_df[forecast_df["is_forecast"] == False]

        lateness_table = build_lateness_table(
            history_rows,
            cutoff_date=start
        )

        # ------------------------------------------------------------
        # Machine-learned miss model (ml_risk_services.py), trained on
        # every past commit received before the forecast start. The
        # inputs it will be given are as stale as the history is, so it
        # is trained on inputs with the same gap before the commit date.
        # ------------------------------------------------------------
        if risk_engine.RISK_BASIS == "model":

            window_dates = commits_df.loc[
                (commits_df["Commit Dt by Suppl"] >= start) &
                (commits_df["Commit Dt by Suppl"] <= end),
                "Commit Dt by Suppl"
            ]

            miss_model = train_miss_model(
                history_rows,
                start,
                choose_lead_days(history_rows, window_dates, start)
            )

    else:
        print(
            "WARNING: 'Working Days Late' is not in the forecast data, so "
            "historical lateness cannot be used. Risk falls back to the "
            "quantity forecast."
        )

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

    # The commit file's own Vendor Name (only present if the user ticked it).
    # Kept under its own column name so the Vendor Name merge below cannot
    # rename or overwrite it. Used only to match the Owner Matrix.
    commits_agg["_Commit Vendor Name"] = (
        commits_agg["Vendor Name"]
        if "Vendor Name" in commits_agg.columns
        else np.nan
    )

    # Number of commits rolled into each Material + date bucket, so the miss
    # model can be given an average commit size (it was trained on single commits).
    commits_agg["_Commit Count"] = (
        commits_window
        .groupby(["Material", "Commit Dt by Suppl"])
        .size()
        .to_numpy()
    )

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

    if "Vendor Name" in forecast_window.columns:
        forecast_window = forecast_window.assign(
            **{"_History Vendor": forecast_window["Vendor Name"]}
        )

    # On-time probability from the vendor's history of late deliveries.
    # Looked up by the same Material + Vendor Name the forecast is labeled with.
    if lateness_table is not None and "Vendor Name" in forecast_window.columns:
        forecast_window = add_on_time_probability(
            forecast_window,
            lateness_table
        )

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

    def norm_name(x):
        if pd.isna(x):
            return ""

        return " ".join(str(x).upper().split())

    # Supplier # is only mapped when the user matches on Vendor Code.
    if "Supplier #" not in owner_df.columns:
        owner_df["Supplier #"] = ""

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

    # A blank Supplier # must never match a blank vendor code on a commit.
    supplier_to_name.pop("", None)

    # ------------------------------------------------------------
    # Supplier -> Buyer
    # ------------------------------------------------------------

    supplier_to_buyer = dict(
        zip(
            owner_df["Supplier #"],
            owner_df["Assigned Buyer"]
        )
    )
    supplier_to_buyer.pop("", None)

    # ------------------------------------------------------------
    # Vendor Name -> Buyer (used only when the commit file's
    # Vendor Name was ticked; the Owner Matrix needs a Supplier Name
    # column mapped for this to match anything)
    # ------------------------------------------------------------

    name_to_buyer = dict(
        zip(
            supplier_names.apply(norm_name),
            owner_df["Assigned Buyer"]
        )
    )
    name_to_buyer.pop("", None)

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

    # Match on Vendor Code first (if ticked), then on Vendor Name (if ticked).
    # If neither was ticked there is nothing to match on -> UNKNOWN.
    assigned_buyer = merged["Vendor"].map(supplier_to_buyer)

    assigned_buyer = assigned_buyer.fillna(
        merged["_Commit Vendor Name"]
        .apply(norm_name)
        .map(name_to_buyer)
    )

    merged["Assigned Buyer"] = assigned_buyer.fillna("UNKNOWN")

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
    # MACHINE-LEARNED CHANCE EACH COMMIT IS MET
    # ============================================================

    if miss_model is not None and "_History Vendor" in merged.columns and not merged.empty:

        merged["Model On-Time Probability"] = predict_on_time_probability(
            miss_model,
            pd.DataFrame({
                "Material": merged["Material"],
                "Vendor Name": merged["_History Vendor"],
                "Commit Date": merged["Commit Dt by Suppl"],
                "Commit Qty": merged["Commit Qty"] / merged["_Commit Count"].clip(lower=1),
            })
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

    # Lateness evidence and the quantity result, when available.
    # Probability / Risk are based on whichever RISK_BASIS selects in risk_engine.py
    # (model, then lateness, then quantity, for any row without a value).
    extra_cols = [
        c for c in (
            "Model On-Time Probability",
            "On-Time Probability",
            "Quantity Probability",
            "On-Time Sample Size"
        )
        if c in merged.columns
    ]

    final = pd.concat([final, merged.loc[final.index, extra_cols]], axis=1)

    return final