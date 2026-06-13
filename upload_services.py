import pandas as pd
import numpy as np
from io import BytesIO

from data_processing import load_historical_file
from forecast_build_services import build_forecast


def process_forecast_uploads(
    hist_file,
    commit_file,
    owner_file,
    new_otd_file,
    forecast_start_date,
    forecast_horizon
):

    # ---------------------------
    # 1. LOAD HISTORICAL
    df = load_historical_file(hist_file)

    # ---------------------------
    # 2. LOAD NEW OTD FILE
    if new_otd_file.filename.lower().endswith(".csv"):
        new_otd_df = pd.read_csv(new_otd_file)
    else:
        new_otd_df = pd.read_excel(new_otd_file)

    new_otd_df.columns = new_otd_df.columns.str.strip()

    print("New OTD columns:")
    print(new_otd_df.columns.tolist())

    # ---------------------------
    # 3. CLEAN + MAP OTD → HISTORICAL FORMAT

    # remove leading zeros in vendor id
    new_otd_df["VENDOR_ID"] = (
        new_otd_df["VENDOR_ID"]
        .astype(str)
        .str.lstrip("0")
    )

    otd_hist_df = pd.DataFrame({
    "Material": new_otd_df["Material"],
    "Vendor Name": new_otd_df["VENDOR_NAME"],
    "Vendor": new_otd_df["VENDOR_ID"],
    "Date Rcvd": new_otd_df["Delivered in Full Date"],
    "Date Due": new_otd_df["Corrected Stat Date"],
    "Qty Due": new_otd_df["Received Qty"],
    "Qty Rcvd": new_otd_df["Received Qty"],
    "Days Late": new_otd_df["DAYS DIFF"],
    "Description": np.select(
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

    # ensure correct types (match historical expectations)
    otd_hist_df["Material"] = otd_hist_df["Material"].astype(str).str.strip()
    otd_hist_df["Date Rcvd"] = pd.to_datetime(otd_hist_df["Date Rcvd"], errors="coerce")
    otd_hist_df["Date Due"] = pd.to_datetime(otd_hist_df["Date Due"], errors="coerce")
    otd_hist_df["Days Late"] = pd.to_numeric(otd_hist_df["Days Late"], errors="coerce")

    # ---------------------------
    # 4. APPEND OTD TO HISTORICAL
    df = pd.concat([df, otd_hist_df], ignore_index=True)

    # ---------------------------
    # 5. RUN FORECAST
    combined, accuracy = build_forecast(
        df,
        forecast_start_date,
        forecast_horizon
    )

    # ---------------------------
    # 6. FORMAT FORECAST OUTPUT
    out = BytesIO()
    combined.to_csv(out, index=False)
    out.seek(0)
    forecast_csv = out.read()

    # ---------------------------
    # 7. LOAD & PROCESS COMMITS
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
    # 8. LOAD OWNER MATRIX
    if owner_file.filename.lower().endswith(".csv"):
        owner_df = pd.read_csv(owner_file)
    else:
        owner_df = pd.read_excel(owner_file)

    out3 = BytesIO()
    owner_df.to_csv(out3, index=False)
    out3.seek(0)
    owner_matrix_csv = out3.read()

    # ---------------------------
    # 9. RETURN OUTPUTS
    return (
        forecast_csv,
        commits_csv,
        owner_matrix_csv,
        accuracy
    )