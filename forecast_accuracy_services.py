import pandas as pd
import numpy as np
from model import xgboost_forecast_with_uncertainty

def calculate_forecast_accuracy(df, forecast_horizon):
    # ---------------------------
    # FORECAST ACCURACY CALCULATION

    df = df.dropna(subset=["Date Received"])
    df["Date Received"] = pd.to_datetime(df["Date Received"], errors="coerce")
    df = df.dropna(subset=["Date Received"])
    
    cutoff = df["Date Received"].max() - pd.Timedelta(days=forecast_horizon)
    train = df[df["Date Received"] <= cutoff]
    test = df[df["Date Received"] > cutoff]

    acc_df = pd.DataFrame()
    if not test.empty:
        acc_dates = pd.date_range(test["Date Received"].min(), test["Date Received"].max(), freq="D")
        acc_parts = []
        for mat, g in sorted(train.groupby("Material"), key=lambda x: x[0]):
            try:
                preds, sigma = xgboost_forecast_with_uncertainty(
                        g["Quantity Received"],
                        g["Date Received"],
                        acc_dates,
                        g["Days Late Classification"],
                        g["Vendor Name"]
                    )
            except Exception as e:
                print(f"❌ FORECAST FAILED | Material={mat} | Error={e}")
                continue
            acc_parts.append(pd.DataFrame({
                "Material": [mat] * len(acc_dates),
                "Date Received": acc_dates,
                "Quantity Received": preds
            }))
        if acc_parts:
            acc_df = pd.concat(acc_parts, ignore_index=True)

    merged = acc_df.merge(
        test[['Material', 'Date Received', 'Quantity Received']],
        on=['Material', 'Date Received'],
        how='inner',
        suffixes=('_pred', '_true')
    )

    if merged.empty:
        accuracy = {
            "AVG_COMMIT_QTY": "N/A",
            "MAE": "N/A",
            "RMSE": "N/A",
            "MAPE": "N/A",
            "R2": "N/A"
        }
    else:
        y_true = merged['Quantity Received_true'].astype(float)
        y_pred = merged['Quantity Received_pred'].astype(float)
        errors = np.maximum(y_true - y_pred, 0)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
        accuracy = {
            "AVG_COMMIT_QTY": round(y_true.mean(), 2),
            "MAE": round(np.mean(errors), 2),
            "RMSE": round(np.sqrt(np.mean(errors**2)), 2),
            "MAPE": f"{(np.mean(errors / y_true.clip(lower=1)) * 100):.2f}%",
            "R2": round(r2, 4)
        }

    return acc_df, accuracy