import pandas as pd
import numpy as np
from scipy.stats import norm

# ---------------------------
# Core logic: evaluate a single row
def evaluate_risk_row(mu, sigma, commit_qty):
    """
    Evaluate probability, risk label, confidence interval and confidence level
    for a single forecast vs commit quantity row.
    """
    # Probability
    if pd.isna(mu) or pd.isna(sigma) or pd.isna(commit_qty):
        prob_pct = 0
        risk = "LOW"
        ci_str = "NA"
        conf_level = "LOW"
    else:
        if sigma == 0:
            prob = 0.5
        else:
            prob = 1 - norm.cdf(commit_qty, loc=mu, scale=sigma)

        prob_pct = round(prob * 100)

        # Risk classification
        if prob_pct < 75:
            risk = "HIGH"
        else:
            risk = "LOW"

        # Confidence interval
        lower = max(mu - 1.96 * sigma, 0)
        upper = mu + 1.96 * sigma
        ci_str = f"{round(lower,2)} - {round(upper,2)}"

        # Confidence level based on interval width
        width = upper - lower
        if width <= 5:
            conf_level = "HIGH"
        elif width <= 15:
            conf_level = "MED"
        else:
            conf_level = "LOW"

    return f"{prob_pct}%", risk, ci_str, conf_level

# ---------------------------
# Apply to entire DataFrame
def apply_risk_model(df):
    """
    Adds Probability, Risk, Confidence Interval, and Confidence columns
    to a dataframe with columns: 'Quantity Received', 'Sigma', 'Commit Qty'.
    """
    # Ensure numeric types
    cols = ["Quantity Received", "Sigma", "Commit Qty"]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    results = df.apply(
        lambda r: evaluate_risk_row(r["Quantity Received"], r["Sigma"], r["Commit Qty"]),
        axis=1
    )

    df[["Probability", "Risk", "Confidence Interval", "Confidence"]] = pd.DataFrame(
        results.tolist(), index=df.index
    )
    return df