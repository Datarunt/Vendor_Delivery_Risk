import pandas as pd
import numpy as np
 
from model import xgboost_forecast_with_uncertainty
from forecast_accuracy_services import calculate_forecast_accuracy
 
# ============================================================
# HELPERS
# ============================================================
 
def clean_columns(df):
    """
    Standardise dataframe column names.
    """
 
    df = df.copy()
 
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
    Validate required columns before they are used.
    """
 
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]
 
    if missing:
 
        raise ValueError(
            f"{dataframe_name} is missing "
            f"required column(s): {missing}. "
            f"Available columns: "
            f"{df.columns.tolist()}"
        )
 
def debug_dataframe(
    df,
    name
):
    """
    Print useful debugging information.
    """
 
    print()
    print("----------------------------------------")
    print(name)
    print("----------------------------------------")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
 
    if "Material" in df.columns:
 
        print(
            "Material exists: True"
        )
 
        print(
            "Material null count:",
            df["Material"].isna().sum()
        )
 
        print(
            "Material sample:",
            df["Material"].head(5).tolist()
        )
 
    else:
 
        print(
            "Material exists: FALSE"
        )
 
# ============================================================
# BUILD FORECAST
# ============================================================
 
def build_forecast(
    df,
    forecast_start_date,
    forecast_horizon
):
 
    print()
    print("################################################")
    print("# BUILD FORECAST")
    print("################################################")
 
    # ========================================================
    # 1. COPY AND CLEAN INPUT
    # ========================================================
 
    df = df.copy()
 
    df = clean_columns(
        df
    )
 
    debug_dataframe(
        df,
        "INPUT TO build_forecast()"
    )
 
    # ========================================================
    # 2. VALIDATE REQUIRED COLUMNS
    # ========================================================
 
    required_columns = [
        "Material",
        "Vendor Name",
        "Date Received",
        "Quantity Received",
        "Days Late Classification",
        "Avg Days Late"
    ]
 
    validate_columns(
        df,
        required_columns,
        "build_forecast input"
    )
 
    # ========================================================
    # 3. NORMALISE MATERIAL
    # ========================================================
 
    df["Material"] = (
        df["Material"]
        .astype(str)
        .str.strip()
    )
 
    # Don't allow the string "nan" to become a material
    df.loc[
        df["Material"].str.lower().eq("nan"),
        "Material"
    ] = np.nan
 
    # ========================================================
    # 4. NORMALISE VENDOR
    # ========================================================
 
    df["Vendor Name"] = (
        df["Vendor Name"]
        .astype(str)
        .str.strip()
    )
 
    # ========================================================
    # 5. NORMALISE DATES
    # ========================================================
 
    df["Date Received"] = pd.to_datetime(
        df["Date Received"],
        errors="coerce"
    )
 
    # ========================================================
    # 6. NORMALISE QUANTITY
    # ========================================================
 
    df["Quantity Received"] = pd.to_numeric(
        df["Quantity Received"],
        errors="coerce"
    )
 
    # ========================================================
    # 7. NORMALISE DAYS LATE
    # ========================================================
 
    df["Avg Days Late"] = pd.to_numeric(
        df["Avg Days Late"],
        errors="coerce"
    )
 
    # ========================================================
    # 8. VALIDATE AFTER NORMALISATION
    # ========================================================
 
    debug_dataframe(
        df,
        "AFTER NORMALISATION"
    )
 
    # Remove rows with no Material
    df = df[
        df["Material"].notna()
    ].copy()
 
    if df.empty:
 
        raise ValueError(
            "No valid rows remain after filtering "
            "blank Material values."
        )
 
    # ========================================================
    # 9. SORT
    # ========================================================
 
    df = df.sort_values(
        [
            "Material",
            "Date Received"
        ]
    ).copy()
 
    print()
    print(
        "Unique materials:",
        df["Material"].nunique()
    )
 
    print(
        "Materials:",
        df["Material"].dropna().unique()[:20]
    )
 
    # ========================================================
    # 10. LATEST MATERIAL ATTRIBUTES
    # ========================================================
 
    print()
    print(
        "STEP: Building latest material attributes"
    )
 
    try:
 
        latest_attrs = (
            df
            .groupby(
                "Material",
                as_index=False
            )
            .last()
        )
 
    except KeyError as e:
 
        raise ValueError(
            f"Failed while grouping latest material "
            f"attributes. Missing column: {e}. "
            f"Available columns: "
            f"{df.columns.tolist()}"
        ) from e
 
    debug_dataframe(
        latest_attrs,
        "LATEST MATERIAL ATTRIBUTES"
    )
 
    # ========================================================
    # 11. VENDOR MAP
    # ========================================================
 
    try:
 
        vendor_map = (
            latest_attrs
            .set_index("Material")[
                "Vendor Name"
            ]
            .to_dict()
        )
 
    except KeyError as e:
 
        raise ValueError(
            f"Failed creating vendor map. "
            f"Missing column: {e}. "
            f"Latest attributes columns: "
            f"{latest_attrs.columns.tolist()}"
        ) from e
 
    # ========================================================
    # 12. ACCURACY
    # ========================================================
 
    print()
    print(
        "STEP: Calculating forecast accuracy"
    )
 
    debug_dataframe(
        df,
        "DATAFRAME SENT TO calculate_forecast_accuracy()"
    )
 
    try:
 
        acc_df, accuracy = (
            calculate_forecast_accuracy(
                df,
                forecast_horizon
            )
        )
 
    except KeyError as e:
 
        raise ValueError(
            f"calculate_forecast_accuracy() "
            f"attempted to access missing column "
            f"{e}. "
            f"Columns available: "
            f"{df.columns.tolist()}"
        ) from e
 
    except Exception as e:
 
        raise ValueError(
            f"calculate_forecast_accuracy() failed: "
            f"{e}. "
            f"Columns available: "
            f"{df.columns.tolist()}"
        ) from e
 
    print(
        "Accuracy calculation completed."
    )
 
    # ========================================================
    # 13. FUTURE DATES
    # ========================================================
 
    print()
    print(
        "STEP: Creating future forecast dates"
    )
 
    future_dates = pd.date_range(
        start=forecast_start_date,
        periods=forecast_horizon,
        freq="D"
    )
 
    print(
        "Forecast start:",
        forecast_start_date
    )
 
    print(
        "Forecast horizon:",
        forecast_horizon
    )
 
    print(
        "Future dates:",
        future_dates.tolist()
    )
 
    # ========================================================
    # 14. FORECAST EACH MATERIAL / VENDOR
    # ========================================================
 
    final_parts = []
 
    grouped = df.groupby(
        [
            "Material",
            "Vendor Name"
        ]
    )
 
    print()
    print(
        "Number of Material/Vendor groups:",
        len(grouped)
    )
 
    for (
        mat,
        vendor
    ), g in sorted(
        grouped,
        key=lambda x: (
            str(x[0][0]),
            str(x[0][1])
        )
    ):
 
        print()
        print(
            "Forecasting Material:",
            mat,
            "| Vendor:",
            vendor,
            "| Rows:",
            len(g)
        )
 
        # ----------------------------------------------------
        # Make sure the group contains usable data
        # ----------------------------------------------------
 
        g = g.copy()
 
        g = g.sort_values(
            "Date Received"
        )
 
        g = g[
            g["Date Received"].notna()
        ].copy()
 
        g = g[
            g["Quantity Received"].notna()
        ].copy()
 
        if g.empty:
 
            print(
                "Skipping group because there "
                "is no usable historical data."
            )
 
            continue
 
        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------
 
        try:
 
            preds, sigma = (
                xgboost_forecast_with_uncertainty(
 
                    g["Quantity Received"],
 
                    g["Date Received"],
 
                    future_dates,
 
                    g["Days Late Classification"],
 
                    g["Avg Days Late"]
                )
            )
 
        except Exception as e:
 
            # DO NOT silently swallow errors.
            print()
            print(
                "FORECAST ERROR"
            )
 
            print(
                "Material:",
                mat
            )
 
            print(
                "Vendor:",
                vendor
            )
 
            print(
                "Error:",
                repr(e)
            )
 
            print(
                "Group columns:",
                g.columns.tolist()
            )
 
            # Continue with other material/vendor groups
            continue
 
        # ----------------------------------------------------
        # Create forecast dataframe
        # ----------------------------------------------------
 
        forecast_part = pd.DataFrame({
 
            "Material":
                mat,
 
            "Date Received":
                future_dates,
 
            "Quantity Received":
                preds,
 
            "Vendor Name":
                vendor_map.get(
                    mat,
                    vendor
                ),
 
            "is_forecast":
                True,
 
            "Sigma":
                sigma
        })
 
        final_parts.append(
            forecast_part
        )
 
    # ========================================================
    # 15. MAKE SURE WE GENERATED SOMETHING
    # ========================================================
 
    print()
    print(
        "Forecast groups generated:",
        len(final_parts)
    )
 
    if not final_parts:
 
        raise ValueError(
            "No forecast data was generated. "
            "Every Material/Vendor group failed "
            "or contained insufficient data."
        )
 
    # ========================================================
    # 16. COMBINE FORECAST DATA
    # ========================================================
 
    forecast_df = pd.concat(
        final_parts,
        ignore_index=True
    )
 
    forecast_df = clean_columns(
        forecast_df
    )
 
    debug_dataframe(
        forecast_df,
        "FORECAST DATAFRAME"
    )
 
    # ========================================================
    # 17. MARK HISTORICAL DATA
    # ========================================================
 
    df["is_forecast"] = False
 
    df["Sigma"] = np.nan
 
    # ========================================================
    # 18. COMBINE HISTORICAL + FORECAST
    # ========================================================
 
    print()
    print(
        "STEP: Combining historical and forecast data"
    )
 
    combined = pd.concat(
        [
            df,
            forecast_df
        ],
        ignore_index=True,
        sort=False
    )
 
    combined = combined.sort_values(
        [
            "Material",
            "Date Received"
        ]
    ).reset_index(
        drop=True
    )
 
    # ========================================================
    # 19. FINAL VALIDATION
    # ========================================================
 
    validate_columns(
        combined,
        [
            "Material",
            "Date Received",
            "Quantity Received",
            "Vendor Name"
        ],
        "Final combined forecast"
    )
 
    debug_dataframe(
        combined,
        "FINAL COMBINED FORECAST"
    )
 
    print()
    print(
        "BUILD FORECAST COMPLETE"
    )
 
    print(
        "Final shape:",
        combined.shape
    )
 
    print(
        "Final Material count:",
        combined["Material"].nunique()
    )
 
    return combined, accuracy

