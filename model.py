import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def xgboost_forecast_with_uncertainty(train,dates,future_dates,descriptions,vendor_names):
    if len(train) < 5:
        raise ValueError("Insufficient data")

     # ---------------------------
    # 1. Description Encoding
    DESC_SCORE = {
        "Hit / On Time": 0,
        "Miss / 1 Day Late": 1,
        "Miss / 2-4 Calendar Days": 3,
        "Miss / 5-15 Calendar Days": 10,
        "Miss / > 15 Calendar Days": 20
    }

    desc_numeric = descriptions.map(DESC_SCORE).fillna(0).astype(float)
    desc_mean = desc_numeric.mean()

    # ---------------------------
    # 2. Vendor Encoding (NEW)
    vendor_encoded, vendor_uniques = pd.factorize(
        vendor_names.astype(str).str.upper().str.strip()
    )

    vendor_encoded = vendor_encoded.astype(float)
    future_vendor = np.full(len(future_dates), vendor_encoded.mean())
      
    # ---------------------------
    # 3. Seasonality Encoding
    months = np.array([d.month for d in dates])

    month_dummies_df = pd.get_dummies(months, drop_first=True)
    month_dummies = month_dummies_df.values

    # ---------------------------
    # 4. Feature Matrix
    base_date = min(dates)

    X = np.column_stack([
        [d.toordinal() - base_date.toordinal() for d in dates],  # trend
        desc_numeric.values,                                   # description
        vendor_encoded,                                        # vendor feature
        month_dummies                                          # seasonality
    ])

    y = train.values.astype(float)

    # ---------------------------
    # 4. XGBoost Regression 
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X, y)

    y_hat = model.predict(X)
    sigma = (y - y_hat).std(ddof=1)

    # ---------------------------
    # 5. Future Feature Matrix
    future_months = np.array([d.month for d in future_dates])

    future_month_dummies = pd.get_dummies(
        future_months,
        drop_first=True
    )

    future_month_dummies = future_month_dummies.reindex(
        columns=month_dummies_df.columns,
        fill_value=0
    ).values

    X_future = np.column_stack([
        [d.toordinal() - base_date.toordinal() for d in future_dates],
        np.full(len(future_dates), desc_mean),
        np.full(len(future_dates), vendor_encoded.mean()),
        future_month_dummies
    ])
    
    preds = model.predict(X_future)

    return np.maximum(preds, 0), sigma
