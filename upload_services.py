import pandas as pd
import numpy as np
from io import BytesIO
 
from data_processing import load_historical_file
from forecast_build_services import build_forecast
 
# ============================================================
# HELPERS
# ============================================================
 
def clean_columns(df):
    """
    Standardise dataframe column names.
    """
 
    if df is None:
        return df
 
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )
 
    return df
 
def validate_columns(
    df,
    required_columns,
    dataframe_name
):
    """
    Validate that required columns exist.
 
    Raises a useful error instead of allowing a later
    pandas KeyError such as:
 
        KeyError: 'Material'
    """
 
    if df is None:
        raise ValueError(
            f"{dataframe_name} dataframe is None."
        )
 
    df = clean_columns(df)
 
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]
 
    if missing:
 
        raise ValueError(
            f"{dataframe_name} is missing required "
            f"column(s): {missing}. "
            f"Available columns are: "
            f"{df.columns.tolist()}"
        )
 
    return df
 
def print_dataframe_info(
    df,
    dataframe_name
):
    """
    Print useful debugging information.
    """
 
    print()
    print("========================================")
    print(dataframe_name)
    print("========================================")
 
    if df is None:
        print("DataFrame is None")
        return
 
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
 
    print(
        "Material exists:",
        "Material" in df.columns
    )
 
    if "Material" in df.columns:
 
        print(
            "Material null count:",
            df["Material"].isna().sum()
        )
 
        print(
            "Material sample:",
            df["Material"].head(5).tolist()
        )
 
def reset_file_stream(file):
    """
    Reset a Werkzeug FileStorage stream.
    """
 
    if file is not None:
 
        try:
            file.stream.seek(0)
        except Exception:
            pass
 
def read_uploaded_dataframe(file):
    """
    Read an uploaded CSV/XLS/XLSX into pandas.
    """
 
    if file is None:
        raise ValueError("No file was supplied.")
 
    reset_file_stream(file)
 
    filename = file.filename.lower()
 
    if filename.endswith(".csv"):
 
        last_error = None
 
        for encoding in (
            "utf-8",
            "cp1252",
            "latin1"
        ):
 
            try:
 
                reset_file_stream(file)
 
                df = pd.read_csv(
                    file.stream,
                    encoding=encoding
                )
 
                return clean_columns(df)
 
            except UnicodeDecodeError as e:
 
                last_error = e
 
        raise ValueError(
            f"Could not decode CSV file "
            f"'{file.filename}'. "
            f"Last error: {last_error}"
        )
 
    elif filename.endswith(
        (".xlsx", ".xls")
    ):
 
        reset_file_stream(file)
 
        df = pd.read_excel(
            file.stream
        )
 
        return clean_columns(df)
 
    else:
 
        raise ValueError(
            f"Unsupported file type: "
            f"{file.filename}"
        )
 
# ============================================================
# MAIN PROCESS
# ============================================================
 
def process_forecast_uploads(
    hist_file,
    commit_file,
    owner_file,
    new_otd_file,
    forecast_start_date,
    forecast_horizon
):
 
    print()
    print("################################################")
    print("# PROCESS FORECAST UPLOADS")
    print("################################################")
 
    # ========================================================
    # 1. LOAD HISTORICAL
    # ========================================================
 
    if hist_file and hist_file.filename:
 
        print()
        print("STEP 1: Loading Historical file")
 
        reset_file_stream(hist_file)
 
        try:
 
            df = load_historical_file(
                hist_file
            )
 
        except Exception as e:
 
            raise ValueError(
                f"Error loading Historical Commits "
                f"file: {e}"
            ) from e
 
        df = clean_columns(df)
 
    else:
 
        df = pd.DataFrame(
            columns=[
                "Vendor Name",
                "Material",
                "Date Received",
                "Quantity Due",
                "Quantity Received",
                "Days Late Classification",
                "Number of Days Late"
            ]
        )
 
    # ========================================================
    # DEBUG HISTORICAL RESULT
    # ========================================================
 
    print_dataframe_info(
        df,
        "HISTORICAL DATA AFTER load_historical_file()"
    )
 
    # ========================================================
    # IMPORTANT:
    # Validate Material BEFORE build_forecast()
    # ========================================================
 
    validate_columns(
        df,
        ["Material"],
        "Historical data returned by load_historical_file()"
    )
 
    # ========================================================
    # Normalise Material
    # ========================================================
 
    df["Material"] = (
        df["Material"]
        .astype(str)
        .str.strip()
    )
 
    # Treat string "nan" as missing
    df.loc[
        df["Material"].str.lower().eq("nan"),
        "Material"
    ] = np.nan
 
    # ========================================================
    # 2. LOAD NEW OTD FILE
    # ========================================================
 
    if (
        new_otd_file
        and new_otd_file.filename
    ):
 
        print()
        print("STEP 2: Loading New OTD file")
 
        new_otd_df = read_uploaded_dataframe(
            new_otd_file
        )
 
        print_dataframe_info(
            new_otd_df,
            "NEW OTD DATA"
        )
 
        # ----------------------------------------------------
        # Normalise Y/N column
        # ----------------------------------------------------
 
        for col in new_otd_df.columns:
 
            if col.lower().strip() == (
                "delivered in full y/n"
            ):
 
                new_otd_df = (
                    new_otd_df.rename(
                        columns={
                            col:
                            "Delivered in Full Y/N"
                        }
                    )
                )
 
                break
 
        # ----------------------------------------------------
        # Validate required OTD columns
        # ----------------------------------------------------
 
        required_otd_columns = [
            "Material",
            "VENDOR_ID",
            "VENDOR_NAME",
            "Stat Date",
            "Delivered in Full Y/N",
            "Delivered in Full Date",
            "Month of OTD Measure",
            "DateDiff Measured Month",
            "Received Qty"
        ]
 
        missing_otd = [
            col
            for col in required_otd_columns
            if col not in new_otd_df.columns
        ]
 
        if missing_otd:
 
            raise ValueError(
                "New OTD file is missing required "
                f"column(s): {missing_otd}. "
                "Available columns: "
                f"{new_otd_df.columns.tolist()}"
            )
 
        # ----------------------------------------------------
        # Normalise vendor ID
        # ----------------------------------------------------
 
        new_otd_df["VENDOR_ID"] = (
            new_otd_df["VENDOR_ID"]
            .astype(str)
            .str.strip()
            .str.lstrip("0")
        )
 
        # ----------------------------------------------------
        # Calculate DAYS DIFF
        # ----------------------------------------------------
 
        def calculate_days_diff(row):
 
            stat_date = pd.to_datetime(
                row["Stat Date"],
                errors="coerce"
            )
 
            delivered = (
                str(
                    row[
                        "Delivered in Full Y/N"
                    ]
                )
                .strip()
                .upper()
            )
 
            # ------------------------------------------------
            # Delivered in full
            # ------------------------------------------------
 
            if delivered in (
                "TRUE",
                "YES",
                "Y",
                "1"
            ):
 
                delivered_date = pd.to_datetime(
                    row[
                        "Delivered in Full Date"
                    ],
                    errors="coerce"
                )
 
                if (
                    pd.isna(delivered_date)
                    or pd.isna(stat_date)
                ):
 
                    return 0
 
                return (
                    delivered_date
                    - stat_date
                ).days
 
            # ------------------------------------------------
            # Not delivered in full
            # ------------------------------------------------
 
            month_of_measure = pd.to_datetime(
                row[
                    "Month of OTD Measure"
                ],
                errors="coerce"
            )
 
            if pd.notna(
                row["DateDiff Measured Month"]
            ):
 
                try:
 
                    n = int(
                        row[
                            "DateDiff Measured Month"
                        ]
                    )
 
                except (
                    ValueError,
                    TypeError
                ):
 
                    n = 1
 
            else:
 
                n = 1
 
            if (
                pd.isna(month_of_measure)
                or pd.isna(stat_date)
            ):
 
                return 0
 
            # ------------------------------------------------
            # Calculate month end
            # ------------------------------------------------
 
            if month_of_measure.month == 12:
 
                month_end = (
                    month_of_measure
                    .replace(day=31)
                )
 
            else:
 
                month_end = (
                    month_of_measure
                    .replace(
                        month=(
                            month_of_measure.month
                            + 1
                        ),
                        day=1
                    )
                    - pd.Timedelta(days=1)
                )
 
            days = (
                month_end
                - stat_date
            ).days + 1
 
            # ------------------------------------------------
            # Additional months
            # ------------------------------------------------
 
            for i in range(1, n):
 
                next_month = (
                    month_end
                    + pd.Timedelta(days=1)
                )
 
                next_month = (
                    next_month
                    + pd.offsets.MonthEnd(i - 1)
                )
 
                if next_month.month == 12:
 
                    next_month_end = (
                        next_month
                        .replace(day=31)
                    )
 
                else:
 
                    next_month_end = (
                        next_month
                        .replace(
                            month=(
                                next_month.month
                                + 1
                            ),
                            day=1
                        )
                        - pd.Timedelta(days=1)
                    )
 
                days += (
                    next_month_end
                    - next_month
                ).days + 1
 
            return days
 
        # ----------------------------------------------------
        # Apply calculation
        # ----------------------------------------------------
 
        new_otd_df["DAYS DIFF"] = (
            new_otd_df.apply(
                calculate_days_diff,
                axis=1
            )
        )
 
        # ----------------------------------------------------
        # Build historical-style OTD dataframe
        # ----------------------------------------------------
 
        otd_hist_df = pd.DataFrame({
 
            "Material":
                new_otd_df["Material"],
 
            "Vendor Name":
                new_otd_df["VENDOR_NAME"],
 
            "Date Received":
                new_otd_df[
                    "Delivered in Full Date"
                ],
 
            "Quantity Due":
                new_otd_df[
                    "Received Qty"
                ],
 
            "Quantity Received":
                new_otd_df[
                    "Received Qty"
                ],
 
            "Number of Days Late":
                new_otd_df[
                    "DAYS DIFF"
                ],
 
            "Days Late Classification":
                np.select(
                    [
                        new_otd_df[
                            "DAYS DIFF"
                        ] < 0,
 
                        new_otd_df[
                            "DAYS DIFF"
                        ] == 0,
 
                        new_otd_df[
                            "DAYS DIFF"
                        ] == 1,
 
                        new_otd_df[
                            "DAYS DIFF"
                        ].between(2, 4),
 
                        new_otd_df[
                            "DAYS DIFF"
                        ].between(5, 15),
 
                        new_otd_df[
                            "DAYS DIFF"
                        ] > 15,
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
 
        otd_hist_df = clean_columns(
            otd_hist_df
        )
 
        # ----------------------------------------------------
        # Normalise OTD Material
        # ----------------------------------------------------
 
        otd_hist_df["Material"] = (
            otd_hist_df["Material"]
            .astype(str)
            .str.strip()
        )
 
        otd_hist_df.loc[
            otd_hist_df["Material"]
            .str.lower()
            .eq("nan"),
            "Material"
        ] = np.nan
 
        # ----------------------------------------------------
        # Dates / numeric values
        # ----------------------------------------------------
 
        otd_hist_df["Date Received"] = (
            pd.to_datetime(
                otd_hist_df[
                    "Date Received"
                ],
                errors="coerce"
            )
        )
 
        otd_hist_df[
            "Number of Days Late"
        ] = pd.to_numeric(
            otd_hist_df[
                "Number of Days Late"
            ],
            errors="coerce"
        )
 
        # ----------------------------------------------------
        # Validate before append
        # ----------------------------------------------------
 
        validate_columns(
            otd_hist_df,
            [
                "Material",
                "Vendor Name",
                "Date Received",
                "Quantity Due",
                "Quantity Received",
                "Number of Days Late",
                "Days Late Classification"
            ],
            "Converted New OTD historical data"
        )
 
        # ----------------------------------------------------
        # Append OTD to historical
        # ----------------------------------------------------
 
        df = pd.concat(
            [
                df,
                otd_hist_df
            ],
            ignore_index=True
        )
 
        df = clean_columns(df)
 
        print_dataframe_info(
            df,
            "HISTORICAL DATA AFTER OTD APPEND"
        )
 
    # ========================================================
    # 3. VALIDATE HISTORICAL DATA
    # ========================================================
 
    print()
    print("STEP 3: Validating historical data")
 
    if df.empty:
 
        raise ValueError(
            "At least one historical file must "
            "be provided and contain data."
        )
 
    validate_columns(
        df,
        ["Material"],
        "Final historical dataframe"
    )
 
    # ========================================================
    # 4. RUN FORECAST
    # ========================================================
 
    print()
    print("STEP 4: Running build_forecast()")
 
    print_dataframe_info(
        df,
        "DATAFRAME BEING SENT TO build_forecast()"
    )
 
    try:
 
        combined, accuracy = build_forecast(
            df,
            forecast_start_date,
            forecast_horizon
        )
 
    except KeyError as e:
 
        raise ValueError(
            f"build_forecast() attempted to access "
            f"a missing column: {e}. "
            f"Columns available immediately before "
            f"build_forecast(): {df.columns.tolist()}"
        ) from e
 
    except Exception as e:
 
        raise ValueError(
            f"build_forecast() failed: {e}. "
            f"Columns available immediately before "
            f"build_forecast(): {df.columns.tolist()}"
        ) from e
 
    # ========================================================
    # Validate forecast output
    # ========================================================
 
    combined = clean_columns(
        combined
    )
 
    print_dataframe_info(
        combined,
        "FORECAST OUTPUT"
    )
 
    # ========================================================
    # 5. FORMAT FORECAST OUTPUT
    # ========================================================
 
    out = BytesIO()
 
    combined.to_csv(
        out,
        index=False
    )
 
    out.seek(0)
 
    forecast_csv = out.read()
 
    # ========================================================
    # 6. LOAD CURRENT COMMITS
    # ========================================================
 
    print()
    print("STEP 6: Loading Current Commits")
 
    commits_df = read_uploaded_dataframe(
        commit_file
    )
 
    print_dataframe_info(
        commits_df,
        "CURRENT COMMITS BEFORE PROCESSING"
    )
 
    # ========================================================
    # Validate Material
    # ========================================================
 
    validate_columns(
        commits_df,
        [
            "Material",
            "Commit Date",
            "Commit Qty"
        ],
        "Current Commits"
    )
 
    # ========================================================
    # Rename fields for combine_services
    # ========================================================
 
    rename_map = {}
 
    if "Commit Date" in commits_df.columns:
 
        rename_map[
            "Commit Date"
        ] = "Commit Dt by Suppl"
 
    if "Vendor Code" in commits_df.columns:
 
        rename_map[
            "Vendor Code"
        ] = "Vendor"
 
    commits_df = commits_df.rename(
        columns=rename_map
    )
 
    # ========================================================
    # Normalise Material
    # ========================================================
 
    commits_df["Material"] = (
        commits_df["Material"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
 
    # ========================================================
    # Normalise Vendor if available
    # ========================================================
 
    if "Vendor" in commits_df.columns:
 
        commits_df["Vendor"] = (
            commits_df["Vendor"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
 
    print_dataframe_info(
        commits_df,
        "CURRENT COMMITS AFTER PROCESSING"
    )
 
    # ========================================================
    # Create commits CSV
    # ========================================================
 
    out2 = BytesIO()
 
    commits_df.to_csv(
        out2,
        index=False
    )
 
    out2.seek(0)
 
    commits_csv = out2.read()
 
    # ========================================================
    # 7. LOAD OWNER MATRIX
    # ========================================================
 
    print()
    print("STEP 7: Loading Owner Matrix")
 
    owner_df = read_uploaded_dataframe(
        owner_file
    )
 
    print_dataframe_info(
        owner_df,
        "OWNER MATRIX"
    )
 
    # ========================================================
    # Owner Matrix does NOT necessarily need Material.
    #
    # Your current mapping is:
    #
    #   Assigned Buyer
    #   Supplier #
    #
    # so validate the fields that are actually expected.
    # ========================================================
 
    if "Assigned Buyer" not in owner_df.columns:
 
        raise ValueError(
            "Owner Matrix is missing 'Assigned Buyer'. "
            f"Available columns: "
            f"{owner_df.columns.tolist()}"
        )
 
    # Supplier # is optional depending on the mapping
    # configuration, so don't force it here.
 
    # ========================================================
    # Create Owner Matrix CSV
    # ========================================================
 
    out3 = BytesIO()
 
    owner_df.to_csv(
        out3,
        index=False
    )
 
    out3.seek(0)
 
    owner_matrix_csv = out3.read()
 
    # ========================================================
    # 8. FINAL VALIDATION
    # ========================================================
 
    print()
    print("========================================")
    print("FINAL OUTPUT VALIDATION")
    print("========================================")
 
    # Forecast
    forecast_check = pd.read_csv(
        BytesIO(forecast_csv)
    )
 
    # Commits
    commits_check = pd.read_csv(
        BytesIO(commits_csv)
    )
 
    # Owner
    owner_check = pd.read_csv(
        BytesIO(owner_matrix_csv)
    )
 
    print(
        "Forecast columns:",
        forecast_check.columns.tolist()
    )
 
    print(
        "Commits columns:",
        commits_check.columns.tolist()
    )
 
    print(
        "Owner columns:",
        owner_check.columns.tolist()
    )
 
    print(
        "Commits Material exists:",
        "Material" in commits_check.columns
    )
 
    # ========================================================
    # 9. RETURN
    # ========================================================
 
    return (
        forecast_csv,
        commits_csv,
        owner_matrix_csv,
        accuracy
    )

