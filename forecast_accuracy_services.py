import pandas as pd
import numpy as np
 
from model import xgboost_forecast_with_uncertainty
 
# ============================================================
# FORECAST ACCURACY
# ============================================================
 
def calculate_forecast_accuracy(
    df,
    forecast_horizon
):
    """
    Calculate forecast accuracy using the most recent
    forecast_horizon days as the test period.
 
    Returns:
        acc_df
        accuracy
    """
 
    print()
    print("========================================")
    print("FORECAST ACCURACY CALCULATION")
    print("========================================")
 
    # --------------------------------------------------------
    # 1. COPY DATA
    # --------------------------------------------------------
 
    df = df.copy()
 
    # --------------------------------------------------------
    # 2. CLEAN COLUMN NAMES
    # --------------------------------------------------------
 
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )
 
    print(
        "Input columns:",
        df.columns.tolist()
    )
 
    # --------------------------------------------------------
    # 3. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------
 
    required_columns = [
        "Material",
        "Date Received",
        "Quantity Received",
        "Days Late Classification",
        "Avg Days Late"
    ]
 
    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]
 
    if missing_columns:
 
        raise ValueError(
            "Forecast accuracy calculation is missing "
            f"required column(s): {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )
 
    # --------------------------------------------------------
    # 4. NORMALISE DATA
    # --------------------------------------------------------
 
    df["Material"] = (
        df["Material"]
        .astype(str)
        .str.strip()
    )
 
    # Remove string "nan"
    df.loc[
        df["Material"].str.lower().eq("nan"),
        "Material"
    ] = np.nan
 
    df["Date Received"] = pd.to_datetime(
        df["Date Received"],
        errors="coerce"
    )
 
    df["Quantity Received"] = pd.to_numeric(
        df["Quantity Received"],
        errors="coerce"
    )
 
    df["Avg Days Late"] = pd.to_numeric(
        df["Avg Days Late"],
        errors="coerce"
    )
 
    # --------------------------------------------------------
    # 5. REMOVE INVALID ROWS
    # --------------------------------------------------------
 
    df = df.dropna(
        subset=[
            "Material",
            "Date Received"
        ]
    ).copy()
 
    if df.empty:
 
        print(
            "No valid historical rows available "
            "for accuracy calculation."
        )
 
        return (
            pd.DataFrame(
                columns=[
                    "Material",
                    "Date Received",
                    "Quantity Received"
                ]
            ),
            {
                "AVG_COMMIT_QTY": "N/A",
                "MAE": "N/A",
                "RMSE": "N/A",
                "MAPE": "N/A",
                "R2": "N/A"
            }
        )
 
    # --------------------------------------------------------
    # 6. CHECK FORECAST HORIZON
    # --------------------------------------------------------
 
    try:
 
        forecast_horizon = int(
            forecast_horizon
        )
 
    except (
        ValueError,
        TypeError
    ):
 
        forecast_horizon = 14
 
    if forecast_horizon <= 0:
 
        forecast_horizon = 14
 
    # --------------------------------------------------------
    # 7. DETERMINE TRAIN / TEST PERIOD
    # --------------------------------------------------------
 
    max_date = df["Date Received"].max()
 
    cutoff = (
        max_date
        - pd.Timedelta(
            days=forecast_horizon
        )
    )
 
    train = df[
        df["Date Received"] <= cutoff
    ].copy()
 
    test = df[
        df["Date Received"] > cutoff
    ].copy()
 
    print(
        "Max historical date:",
        max_date
    )
 
    print(
        "Cutoff date:",
        cutoff
    )
 
    print(
        "Training rows:",
        len(train)
    )
 
    print(
        "Testing rows:",
        len(test)
    )
 
    # --------------------------------------------------------
    # 8. IMPORTANT:
    # ALWAYS INITIALISE acc_df WITH ITS EXPECTED COLUMNS
    # --------------------------------------------------------
 
    acc_df = pd.DataFrame(
        columns=[
            "Material",
            "Date Received",
            "Quantity Received"
        ]
    )
 
    # --------------------------------------------------------
    # 9. IF THERE IS NO TEST DATA
    # --------------------------------------------------------
 
    if test.empty:
 
        print(
            "No test data available. "
            "Returning N/A accuracy metrics."
        )
 
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
 
        return acc_df, accuracy
 
    # --------------------------------------------------------
    # 10. IF THERE IS NOT ENOUGH TRAINING DATA
    # --------------------------------------------------------
 
    if train.empty:
 
        print(
            "No training data available. "
            "Returning N/A accuracy metrics."
        )
 
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
 
        return acc_df, accuracy
 
    # --------------------------------------------------------
    # 11. ACCURACY FORECAST DATES
    # --------------------------------------------------------
 
    acc_dates = pd.date_range(
        test["Date Received"].min(),
        test["Date Received"].max(),
        freq="D"
    )
 
    print(
        "Accuracy forecast dates:",
        acc_dates.tolist()
    )
 
    # --------------------------------------------------------
    # 12. FORECAST EACH MATERIAL
    # --------------------------------------------------------
 
    acc_parts = []
 
    material_groups = train.groupby(
        "Material"
    )
 
    print(
        "Training material groups:",
        len(material_groups)
    )
 
    for mat, g in sorted(
        material_groups,
        key=lambda x: str(x[0])
    ):
 
        print()
        print(
            f"Accuracy forecast | Material={mat}"
        )
 
        g = g.copy()
 
        # ----------------------------------------------------
        # Sort historical observations
        # ----------------------------------------------------
 
        g = g.sort_values(
            "Date Received"
        )
 
        # ----------------------------------------------------
        # Remove rows without required model inputs
        # ----------------------------------------------------
 
        g = g.dropna(
            subset=[
                "Date Received",
                "Quantity Received"
            ]
        )
 
        if g.empty:
 
            print(
                f"Skipping Material={mat}: "
                "no usable training rows."
            )
 
            continue
 
        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------
 
        try:
 
            preds, sigma = (
                xgboost_forecast_with_uncertainty(
 
                    g["Quantity Received"],
 
                    g["Date Received"],
 
                    acc_dates,
 
                    g["Days Late Classification"],
 
                    g["Avg Days Late"]
                )
            )
 
        except Exception as e:
 
            print(
                f"FORECAST FAILED | "
                f"Material={mat} | "
                f"Error={repr(e)}"
            )
 
            continue
 
        # ----------------------------------------------------
        # Validate prediction length
        # ----------------------------------------------------
 
        if preds is None:
 
            print(
                f"FORECAST FAILED | "
                f"Material={mat} | "
                "Model returned None."
            )
 
            continue
 
        preds = np.asarray(
            preds
        )
 
        if len(preds) != len(acc_dates):
 
            print(
                f"FORECAST FAILED | "
                f"Material={mat} | "
                f"Expected {len(acc_dates)} predictions "
                f"but received {len(preds)}."
            )
 
            continue
 
        # ----------------------------------------------------
        # Create accuracy forecast dataframe
        # ----------------------------------------------------
 
        acc_parts.append(
            pd.DataFrame({
 
                "Material":
                    [mat] * len(acc_dates),
 
                "Date Received":
                    acc_dates,
 
                "Quantity Received":
                    preds
            })
        )
 
    # --------------------------------------------------------
    # 13. COMBINE ACCURACY FORECASTS
    # --------------------------------------------------------
 
    if acc_parts:
 
        acc_df = pd.concat(
            acc_parts,
            ignore_index=True
        )
 
    else:
 
        # IMPORTANT:
        # Keep the required columns even when empty.
        acc_df = pd.DataFrame(
            columns=[
                "Material",
                "Date Received",
                "Quantity Received"
            ]
        )
 
    print()
    print(
        "Accuracy forecast rows:",
        len(acc_df)
    )
 
    print(
        "Accuracy forecast columns:",
        acc_df.columns.tolist()
    )
 
    # --------------------------------------------------------
    # 14. IF NO ACCURACY FORECAST WAS GENERATED
    # --------------------------------------------------------
 
    if acc_df.empty:
 
        print(
            "No accuracy forecasts were generated. "
            "Returning N/A metrics."
        )
 
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
 
        return acc_df, accuracy
 
    # --------------------------------------------------------
    # 15. PREPARE TEST DATA
    # --------------------------------------------------------
 
    test_accuracy = test[
        [
            "Material",
            "Date Received",
            "Quantity Received"
        ]
    ].copy()
 
    test_accuracy["Material"] = (
        test_accuracy["Material"]
        .astype(str)
        .str.strip()
    )
 
    test_accuracy["Date Received"] = (
        pd.to_datetime(
            test_accuracy["Date Received"],
            errors="coerce"
        )
    )
 
    test_accuracy["Quantity Received"] = (
        pd.to_numeric(
            test_accuracy["Quantity Received"],
            errors="coerce"
        )
    )
 
    # --------------------------------------------------------
    # 16. PREPARE PREDICTIONS
    # --------------------------------------------------------
 
    acc_df["Material"] = (
        acc_df["Material"]
        .astype(str)
        .str.strip()
    )
 
    acc_df["Date Received"] = (
        pd.to_datetime(
            acc_df["Date Received"],
            errors="coerce"
        )
    )
 
    acc_df["Quantity Received"] = (
        pd.to_numeric(
            acc_df["Quantity Received"],
            errors="coerce"
        )
    )
 
    # --------------------------------------------------------
    # 17. MERGE PREDICTIONS WITH ACTUALS
    # --------------------------------------------------------
 
    print()
    print(
        "Merging predicted and actual values..."
    )
 
    merged = acc_df.merge(
 
        test_accuracy,
 
        on=[
            "Material",
            "Date Received"
        ],
 
        how="inner",
 
        suffixes=(
            "_pred",
            "_true"
        )
    )
 
    print(
        "Merged rows:",
        len(merged)
    )
 
    # --------------------------------------------------------
    # 18. NO MATCHING ACCURACY DATA
    # --------------------------------------------------------
 
    if merged.empty:
 
        print(
            "No matching prediction/actual "
            "rows were found."
        )
 
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
 
        return acc_df, accuracy
 
    # --------------------------------------------------------
    # 19. CLEAN NUMERIC VALUES
    # --------------------------------------------------------
 
    y_true = pd.to_numeric(
        merged[
            "Quantity Received_true"
        ],
        errors="coerce"
    )
 
    y_pred = pd.to_numeric(
        merged[
            "Quantity Received_pred"
        ],
        errors="coerce"
    )
 
    valid = (
        y_true.notna()
        & y_pred.notna()
    )
 
    y_true = y_true[
        valid
    ]
 
    y_pred = y_pred[
        valid
    ]
 
    # --------------------------------------------------------
    # 20. NO VALID NUMERIC DATA
    # --------------------------------------------------------
 
    if len(y_true) == 0:
 
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
 
        return acc_df, accuracy
 
    # --------------------------------------------------------
    # 21. CALCULATE ERRORS
    # --------------------------------------------------------
 
    raw_errors = (
        y_true - y_pred
    )
 
    # Keep your original behaviour:
    # only count under-forecasting errors.
    errors = np.maximum(
        raw_errors,
        0
    )
 
    # --------------------------------------------------------
    # 22. MAE
    # --------------------------------------------------------
 
    mae = np.mean(
        np.abs(errors)
    )
 
    # --------------------------------------------------------
    # 23. RMSE
    # --------------------------------------------------------
 
    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )
 
    # --------------------------------------------------------
    # 24. MAPE
    # --------------------------------------------------------
 
    denominator = (
        y_true
        .abs()
        .clip(lower=1)
    )
 
    mape = (
        np.mean(
            np.abs(errors)
            / denominator
        )
        * 100
    )
 
    # --------------------------------------------------------
    # 25. R2
    # --------------------------------------------------------
 
    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )
 
    ss_tot = np.sum(
        (
            y_true
            - y_true.mean()
        ) ** 2
    )
 
    if ss_tot == 0:
 
        r2 = 0
 
    else:
 
        r2 = (
            1
            - (
                ss_res
                / ss_tot
            )
        )
 
    # --------------------------------------------------------
    # 26. FINAL ACCURACY DICTIONARY
    # --------------------------------------------------------
 
    accuracy = {
 
        "AVG_COMMIT_QTY":
            round(
                float(y_true.mean()),
                2
            ),
 
        "MAE":
            round(
                float(mae),
                2
            ),
 
        "RMSE":
            round(
                float(rmse),
                2
            ),
 
        "MAPE":
            f"{float(mape):.2f}%",
 
        "R2":
            round(
                float(r2),
                4
            )
    }
 
    print()
    print(
        "========================================"
    )
 
    print(
        "ACCURACY RESULTS"
    )
 
    print(
        "========================================"
    )
 
    print(
        accuracy
    )
 
    print(
        "========================================"
    )
 
    return (
        acc_df,
        accuracy
    )

