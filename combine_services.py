import pandas as pd
from risk_engine import apply_risk_model


# ---------------------------
# COMMIT DATE AGGREGATION
def aggregate_commit_dates(dates, aggregation):
    """
    Buckets each commit date according to the chosen aggregation level.

    - "day"   : no change (current/default behavior)
    - "week"  : rolled forward to the coming Sunday for that date
                (if the date already IS a Sunday, it stays put)
    - "month" : rolled forward to the last calendar day of that month
                (if the date already IS the last day, it stays put)

    Bucketing dates this way means commits that land in the same
    week/month share the same "Commit Dt by Suppl" value, so the
    existing groupby(["Material", "Commit Dt by Suppl"]).sum() below
    naturally combines their Commit Qty together.
    """
    aggregation = (aggregation or "day").strip().lower()

    if aggregation == "week":
        # Monday=0 ... Sunday=6 -> days needed to reach the coming Sunday
        days_until_sunday = (6 - dates.dt.dayofweek) % 7
        return dates + pd.to_timedelta(days_until_sunday, unit="D")

    elif aggregation == "month":
        return dates + pd.offsets.MonthEnd(0)

    # default: "day" (or anything unrecognized) -> unchanged
    return dates


def build_combined_output(forecast_df,commits_df,owner_df,forecast_start_date,forecast_horizon,aggregation="day"):



    # ---------------------------
    # Clean Data
    forecast_df["Material"] = forecast_df["Material"].astype(str).str.strip().str.upper()
    commits_df["Material"] = commits_df["Material"].astype(str).str.strip().str.upper()
    forecast_df["Date Received"] = pd.to_datetime(forecast_df["Date Received"])
    commits_df["Commit Dt by Suppl"] = pd.to_datetime(commits_df["Commit Dt by Suppl"], errors="coerce")
    commits_df["Commit Qty"] = pd.to_numeric(commits_df["Commit Qty"], errors="coerce")

    start = forecast_start_date
    end = start + pd.Timedelta(days=forecast_horizon - 1)

    # Filter to the forecast window FIRST, using each commit's real due
    # date -- then bucket into week/month. This way the window boundary
    # is judged on the actual commit date, not the post-aggregation one.
    commits_window = commits_df[(commits_df["Commit Dt by Suppl"] >= start) &
                                (commits_df["Commit Dt by Suppl"] <= end)].copy()

    commits_window["Commit Dt by Suppl"] = aggregate_commit_dates(
        commits_window["Commit Dt by Suppl"], aggregation
    )

    forecast_window = forecast_df[(forecast_df["Date Received"] >= start) &
                                  (forecast_df["Date Received"] <= end) &
                                  (forecast_df["is_forecast"] == True)].copy()

    # dedup - keep row with lowest sigma (most confident forecast) per day
    forecast_window = forecast_window.sort_values("Sigma").drop_duplicates(
        subset=["Material", "Date Received"], keep="first"
    )

    # Bucket the forecast dates the SAME way the commit dates were bucketed,
    # then combine the daily forecasts that fall in the same bucket:
    #   - Quantity Received (predicted qty) is summed -> total expected
    #     delivery across the whole week/month, matching the summed commit.
    #   - Sigma is combined as sqrt(sum of squares), the standard way to
    #     combine uncertainty across independent daily forecasts, rather
    #     than just keeping one day's sigma.
    # For "day" aggregation this is a no-op: each bucket has exactly one
    # row already, so the sum/combine just returns that same row's values.
    forecast_window["Date Received"] = aggregate_commit_dates(
        forecast_window["Date Received"], aggregation
    )

    forecast_agg_dict = {"Quantity Received": "sum"}
    if "Sigma" in forecast_window.columns:
        forecast_agg_dict["Sigma"] = lambda s: (s.pow(2).sum()) ** 0.5
    if "Vendor Name" in forecast_window.columns:
        forecast_agg_dict["Vendor Name"] = "first"

    forecast_window = forecast_window.groupby(
        ["Material", "Date Received"], as_index=False
    ).agg(forecast_agg_dict)

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