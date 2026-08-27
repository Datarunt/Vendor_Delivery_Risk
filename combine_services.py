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

    # ------------------------------------------------------------

    forecast_window["Date Received"] = aggregate_commit_dates(

        forecast_window["Date Received"],

        aggregation

    )

 

    # ------------------------------------------------------------

    # Aggregate forecasts.

    #

    # Quantity:

    #     sum of daily forecast quantities

    #

    # Sigma:

    #     sqrt(sum of daily sigma^2)

    #

    # IMPORTANT:

    # We keep track of how many valid Sigma values exist.

    # ------------------------------------------------------------

 

    forecast_agg_dict = {

        "Quantity Received": "sum"

    }

 

    if "Sigma" in forecast_window.columns:

 

        # Do NOT replace NaN Sigma with zero.

        forecast_window["_sigma_sq"] = (

            forecast_window["Sigma"] ** 2

        )

 

        forecast_window["_sigma_valid"] = (

            forecast_window["Sigma"].notna().astype(int)

        )

 

        forecast_agg_dict["_sigma_sq"] = "sum"

        forecast_agg_dict["_sigma_valid"] = "sum"

 

    if "Vendor Name" in forecast_window.columns:

        forecast_agg_dict["Vendor Name"] = "first"

 

    forecast_window = forecast_window.groupby(

        ["Material", "Date Received"],

        as_index=False

    ).agg(forecast_agg_dict)

 

    # ------------------------------------------------------------

    # Reconstruct aggregated Sigma

    # ------------------------------------------------------------

    if "Sigma" in forecast_df.columns:

 

        forecast_window["Sigma"] = np.sqrt(

            forecast_window["_sigma_sq"]

        )

 

        # If there were NO valid Sigma observations in the bucket,

        # Sigma must remain NaN.

        forecast_window.loc[

            forecast_window["_sigma_valid"] == 0,

            "Sigma"

        ] = np.nan

 

        forecast_window = forecast_window.drop(

            columns=["_sigma_sq", "_sigma_valid"]

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