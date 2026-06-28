import pandas as pd
import numpy as np

from model import xgboost_forecast_with_uncertainty
from forecast_accuracy_services import calculate_forecast_accuracy


def build_forecast(df, forecast_start_date, forecast_horizon):

    # ---------------------------
    # latest material attrs
    df = df.sort_values(['Material', 'Date Received'])

    latest_attrs = df.groupby('Material', as_index=False).last()

    vendor_map = latest_attrs.set_index('Material')['Vendor Name'].to_dict()

    # ---------------------------
    # accuracy
    acc_df, accuracy = calculate_forecast_accuracy(df, forecast_horizon)

    # ---------------------------
    # future dates
    future_dates = pd.date_range(
        start=forecast_start_date,
        periods=forecast_horizon,
        freq='D'
    )

    final_parts = []

    for (mat, vendor), g in sorted(
        df.groupby(["Material", "Vendor Name"]),
        key=lambda x: (x[0][0], x[0][1])
    ):

        try:
            preds, sigma = xgboost_forecast_with_uncertainty(
                g['Quantity Received'],
                g['Date Received'],
                future_dates,
                g['Days Late Classification'],
                g['Vendor Name']
            )

        except:
            continue

        final_parts.append(pd.DataFrame({
            "Material": mat,
            "Date Received": future_dates,
            "Quantity Received": preds,
            "Vendor Name": vendor_map.get(mat, "UNKNOWN"),
            "is_forecast": True,
            "Sigma": sigma
        }))

    forecast_df = pd.concat(final_parts, ignore_index=True)

    df['is_forecast'] = False
    df['Sigma'] = np.nan

    combined = pd.concat([df, forecast_df]).sort_values(
        ['Material', 'Date Received']
    )

    return combined, accuracy