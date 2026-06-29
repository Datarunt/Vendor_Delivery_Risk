import pandas as pd
import numpy as np
from io import BytesIO

from data_processing import load_historical_file
from forecast_build_services import build_forecast
from werkzeug.datastructures import FileStorage


def process_forecast_uploads(
    hist_file,
    commit_file,
    owner_file,
    new_otd_file,
    forecast_start_date,
    forecast_horizon
):

    # ---------------------------
    # 1. LOAD HISTORICAL (optional)
    if hist_file and hist_file.filename:
        df = load_historical_file(hist_file)
    else:
        df = pd.DataFrame(columns=[
            'Vendor Name', 'Material', 'Date Received', 'Quantity Due',
            'Quantity Received', 'Days Late Classification', 'Number of Days Late'
        ])

    # ---------------------------
    # 2. LOAD NEW OTD FILE (optional)
    if new_otd_file and new_otd_file.filename:
        if new_otd_file.filename.lower().endswith(".csv"):
            new_otd_df = pd.read_csv(new_otd_file)
        else:
            new_otd_df = pd.read_excel(new_otd_file)

        new_otd_df.columns = new_otd_df.columns.str.strip()

        # normalize the Y/N column name regardless of case
        for col in new_otd_df.columns:
            if col.lower() == "delivered in full y/n":
                new_otd_df = new_otd_df.rename(columns={col: "Delivered in Full Y/N"})
                break

        print("New OTD columns:")
        print(new_otd_df.columns.tolist())

        # remove leading zeros in vendor id
        new_otd_df["VENDOR_ID"] = (
            new_otd_df["VENDOR_ID"]
            .astype(str)
            .str.lstrip("0")
        )

        # --------------------------
        # CALCULATE DAYS DIFF
        def calculate_days_diff(row):
            stat_date = pd.to_datetime(row["Stat Date"], errors="coerce")
            delivered = str(row["Delivered in Full Y/N"]).strip().upper()

            if delivered in ("TRUE", "YES", "Y", "1"):
                delivered_date = pd.to_datetime(row["Delivered in Full Date"], errors="coerce")
                if pd.isna(delivered_date) or pd.isna(stat_date):
                    return 0
                return (delivered_date - stat_date).days

            else:
                month_of_measure = pd.to_datetime(row["Month of OTD Measure"], errors="coerce")
                n = int(row["DateDiff Measured Month"]) if pd.notna(row["DateDiff Measured Month"]) else 1

                if pd.isna(month_of_measure) or pd.isna(stat_date):
                    return 0

                if month_of_measure.month == 12:
                    month_end = month_of_measure.replace(day=31)
                else:
                    month_end = month_of_measure.replace(
                        month=month_of_measure.month + 1, day=1
                    ) - pd.Timedelta(days=1)

                days = (month_end - stat_date).days + 1

                for i in range(1, n):
                    next_month = month_end + pd.Timedelta(days=1)
                    next_month = next_month + pd.offsets.MonthEnd(i - 1)
                    if next_month.month == 12:
                        next_month_end = next_month.replace(day=31)
                    else:
                        next_month_end = next_month.replace(
                            month=next_month.month + 1, day=1
                        ) - pd.Timedelta(days=1)
                    days += (next_month_end - next_month).days + 1

                return days

        new_otd_df["DAYS DIFF"] = new_otd_df.apply(calculate_days_diff, axis=1)

        otd_hist_df = pd.DataFrame({
            "Material": new_otd_df["Material"],
            "Vendor Name": new_otd_df["VENDOR_NAME"],
            "Date Received": new_otd_df["Delivered in Full Date"],
            "Quantity Due": new_otd_df["Received Qty"],
            "Quantity Received": new_otd_df["Received Qty"],
            "Number of Days Late": new_otd_df["DAYS DIFF"],
            "Days Late Classification": np.select(
                [
                    new_otd_df["DAYS DIFF"] < 0,
                    new_otd_df["DAYS DIFF"] == 0,
                    new_otd_df["DAYS DIFF"] == 1,
                    new_otd_df["DAYS DIFF"].between(2, 4),
                    new_otd_df["DAYS DIFF"].between(5, 15),
                    new_otd_df["DAYS DIFF"] > 15,
                ],
                [
                    "Miss / Early",
                    "Hit / On Time",
                    "Miss / 1 Day Late",
                    "Miss / 2-4 Calendar Days",
                    "Miss / 5-15 Calendar Days",
                    "Miss / > 15 Calendar Days",
                ],
                default="Hit / On Time"
            )
        })

        otd_hist_df["Material"] = otd_hist_df["Material"].astype(str).str.strip()
        otd_hist_df["Date Received"] = pd.to_datetime(otd_hist_df["Date Received"], errors="coerce")
        otd_hist_df["Number of Days Late"] = pd.to_numeric(otd_hist_df["Number of Days Late"], errors="coerce")

        # append OTD to historical
        df = pd.concat([df, otd_hist_df], ignore_index=True)

    # ---------------------------
    # 3. VALIDATE WE HAVE DATA
    if df.empty:
        raise ValueError("At least one historical file must be provided.")

    # ---------------------------
    # 4. RUN FORECAST
    combined, accuracy = build_forecast(
        df,
        forecast_start_date,
        forecast_horizon
    )

    # ---------------------------
    # 5. FORMAT FORECAST OUTPUT
    out = BytesIO()
    combined.to_csv(out, index=False)
    out.seek(0)
    forecast_csv = out.read()

    # ---------------------------
    # 6. LOAD & PROCESS COMMITS
    if commit_file.filename.lower().endswith(".csv"):
        commits_df = pd.read_csv(commit_file)
    else:
        commits_df = pd.read_excel(commit_file)

    commits_df["Material"] = (
        commits_df["Material"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    commits_df["Vendor"] = (
        commits_df["Vendor"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out2 = BytesIO()
    commits_df.to_csv(out2, index=False)
    out2.seek(0)
    commits_csv = out2.read()

    # ---------------------------
    # 7. LOAD OWNER MATRIX
    if owner_file.filename.lower().endswith(".csv"):
        owner_df = pd.read_csv(owner_file)
    else:
        owner_df = pd.read_excel(owner_file)

    out3 = BytesIO()
    owner_df.to_csv(out3, index=False)
    out3.seek(0)
    owner_matrix_csv = out3.read()

    # ---------------------------
    # 8. RETURN OUTPUTS
    return (
        forecast_csv,
        commits_csv,
        owner_matrix_csv,
        accuracy
    )