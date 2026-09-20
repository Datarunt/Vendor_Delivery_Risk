import pandas as pd
import numpy as np
from scipy.stats import norm

# Confidence thresholds derived empirically from the tertiles of relative
# CI width (width / mu) on the current dataset. Re-derive periodically as
# vendor mix / order sizes shift — these are not universal constants.
CONF_HIGH_THRESHOLD = 0.0356  # <= this -> HIGH confidence
CONF_MED_THRESHOLD = 0.1227   # <= this -> MED confidence, else LOW

# ---------------------------
# What the Risk label is based on.
#
#   "model"     the machine-learned chance the commit is met (on time AND in
#               full), from ml_risk_services.py. HIGH when it is below
#               MODEL_HIGH_RISK_BELOW.
#   "lateness"  the on-time probability from the vendor's / material's plain
#               history of late deliveries (lateness_services.py). HIGH when it
#               is below LATENESS_HIGH_RISK_BELOW.
#   "quantity"  the original behavior: the chance the forecast delivery quantity
#               reaches the commit quantity, HIGH when it is below 75%.
#
# Each basis falls back to the next when it has nothing for a row: model ->
# lateness -> quantity. (For example, the model is skipped when there is too
# little history to train it.) The quantity forecast is still produced either
# way: Quantity Probability, Confidence Interval and Confidence always describe it.
RISK_BASIS = "model"

# Historical on-time rates run high (a typical vendor is on time 90%+ of the
# time), so this cutoff is higher than the 75% used for quantity. 0.90 flags
# roughly the share of commits that turned out to be missed in the saved
# evaluation runs. Re-tune it with performance_check.py as data changes.
LATENESS_HIGH_RISK_BELOW = 0.90

# The model's probabilities are spread wider and sit lower than the scorecard's
# (it also counts short deliveries as misses), so it has its own cutoff. Chosen
# on the saved evaluation runs; re-tune it with performance_check.py.
MODEL_HIGH_RISK_BELOW = 0.85

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

        # Confidence level based on RELATIVE interval width (width / mu),
        # using empirically-derived tertile cut points rather than
        # absolute units or guessed percentages.
        width = upper - lower
        if mu and mu > 0:
            rel_width = width / mu
        else:
            rel_width = float("inf")

        if rel_width <= CONF_HIGH_THRESHOLD:
            conf_level = "HIGH"
        elif rel_width <= CONF_MED_THRESHOLD:
            conf_level = "MED"
        else:
            conf_level = "LOW"

    return f"{prob_pct}%", risk, ci_str, conf_level

# ---------------------------
# Apply to entire DataFrame
def _as_number(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")

    return pd.Series(np.nan, index=df.index)


def _as_pct_text(prob):
    pct = (prob * 100).round()

    return pd.Series(
        np.where(prob.notna(), pct.astype("Int64").astype(str) + "%", "NA"),
        index=prob.index
    )


def apply_risk_model(df):
    """
    Adds Probability, Risk, Confidence Interval, Confidence and Quantity
    Probability columns to a dataframe with columns: 'Quantity Received',
    'Sigma', 'Commit Qty'.

    If the dataframe also has 'Model On-Time Probability' and/or 'On-Time
    Probability' (0-1), Probability and Risk come from them according to
    RISK_BASIS, falling back model -> lateness -> quantity for any row that
    has no value.
    """
    # Ensure numeric types
    cols = ["Quantity Received", "Sigma", "Commit Qty"]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    results = df.apply(
        lambda r: evaluate_risk_row(r["Quantity Received"], r["Sigma"], r["Commit Qty"]),
        axis=1
    )

    quantity = pd.DataFrame(
        results.tolist(),
        index=df.index,
        columns=["Quantity Probability", "Quantity Risk", "Confidence Interval", "Confidence"]
    )

    df["Quantity Probability"] = quantity["Quantity Probability"]
    df["Confidence Interval"] = quantity["Confidence Interval"]
    df["Confidence"] = quantity["Confidence"]

    # Default: the quantity result
    df["Probability"] = quantity["Quantity Probability"]
    df["Risk"] = quantity["Quantity Risk"]

    model_prob = _as_number(df, "Model On-Time Probability")
    lateness_prob = _as_number(df, "On-Time Probability")

    applied = pd.Series(False, index=df.index)

    def use(prob, cutoff):
        rows = prob.notna() & ~applied

        pct = (prob[rows] * 100).round()

        df.loc[rows, "Probability"] = pct.astype(int).astype(str) + "%"
        df.loc[rows, "Risk"] = np.where(pct < cutoff * 100, "HIGH", "LOW")

        applied.loc[rows] = True

    if RISK_BASIS == "model":
        use(model_prob, MODEL_HIGH_RISK_BELOW)

    if RISK_BASIS in ("model", "lateness"):
        use(lateness_prob, LATENESS_HIGH_RISK_BELOW)

    # shown as percentages, like the other probabilities
    for col, prob in (
        ("On-Time Probability", lateness_prob),
        ("Model On-Time Probability", model_prob),
    ):
        if col in df.columns:
            df[col] = _as_pct_text(prob)

    return df