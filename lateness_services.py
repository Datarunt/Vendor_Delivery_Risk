import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# HISTORICAL LATENESS
# ---------------------------------------------------------------------------
# How often has this vendor / material delivered late in the past? That rate
# becomes the on-time probability for a current commit.
#
# The quantity forecast (model.py) says nothing about timing. On the saved
# evaluation runs, a vendor's history of late deliveries separated the commits
# that were actually met from those that were not far better than the quantity
# probability did, and most missed commits were misses on timing: nothing had
# arrived by the commit date.

# A delivery counts as late when it arrives this many working days after the
# commit date, or more. 1 = any lateness. The risk check scores a commit at its
# commit date with no grace period, so 1 is the matching definition. Raise it to
# ignore small slips (2-5 gave similar results on the saved runs).
LATE_AFTER_WORKING_DAYS = 1

# How strongly a thin record is pulled toward the broader rate, measured in
# deliveries. A Material + Vendor pair is blended with its vendor's rate, and a
# vendor with the overall rate. With 20, a pair with 20 deliveries counts its
# own record and its vendor's record equally; a pair with 2 is almost entirely
# its vendor's rate. This stops a material with 3 on-time deliveries from
# looking 100% reliable.
PRIOR_STRENGTH = 20


def build_lateness_table(history, cutoff_date=None):
    """
    Late rates for the whole history, each vendor, and each Material + Vendor pair.

    history      DataFrame with Material, Vendor Name, Date Received and
                 Working Days Late (see data_processing.py)
    cutoff_date  only deliveries received BEFORE this date are used, so a commit
                 is never judged with deliveries that happened after the forecast
                 start

    Returns a dict: global_rate, vendor (late_rate, n), pair (late_rate, n),
    n_rows.
    """
    h = history[["Material", "Vendor Name", "Date Received", "Working Days Late"]].copy()

    h["Date Received"] = pd.to_datetime(h["Date Received"], errors="coerce")
    h["Working Days Late"] = pd.to_numeric(h["Working Days Late"], errors="coerce")
    h = h.dropna(subset=["Date Received", "Working Days Late"])

    used_cutoff = False

    if cutoff_date is not None and not h.empty:
        before = h[h["Date Received"] < pd.to_datetime(cutoff_date)]

        if before.empty:
            print(
                "WARNING: no deliveries in the history before the forecast start "
                "date. Lateness is being measured on ALL deliveries instead."
            )
        else:
            h = before
            used_cutoff = True

    h["late"] = (h["Working Days Late"] >= LATE_AFTER_WORKING_DAYS).astype(int)

    global_rate = h["late"].mean() if len(h) else np.nan

    vendor = h.groupby("Vendor Name")["late"].agg(late_count="sum", n="count")
    vendor["late_rate"] = (
        (vendor["late_count"] + PRIOR_STRENGTH * global_rate)
        / (vendor["n"] + PRIOR_STRENGTH)
    )

    pair = (
        h.groupby(["Material", "Vendor Name"])["late"]
        .agg(late_count="sum", n="count")
    )
    vendor_rate_for_pair = pair.index.get_level_values("Vendor Name").map(
        vendor["late_rate"]
    )
    pair["late_rate"] = (
        (pair["late_count"] + PRIOR_STRENGTH * np.asarray(vendor_rate_for_pair, dtype=float))
        / (pair["n"] + PRIOR_STRENGTH)
    )

    print(
        f"\nLateness history: {len(h)} deliveries"
        + (f" before {pd.to_datetime(cutoff_date).date()}" if used_cutoff else "")
        + f", {len(vendor)} vendors, {len(pair)} Material + Vendor pairs. "
        + (f"Overall late rate {global_rate:.1%}." if len(h) else "")
    )

    return {
        "global_rate": global_rate,
        "vendor": vendor,
        "pair": pair,
        "n_rows": len(h),
    }


def add_on_time_probability(df, table, material_col="Material", vendor_col="Vendor Name"):
    """
    Adds two columns to df (one row per commit or forecast row):

    On-Time Probability   1 - the late rate. Uses the Material + Vendor pair's
                          rate, blended toward its vendor's rate; a pair with no
                          history gets its vendor's rate, and an unknown vendor
                          gets the overall rate.
    On-Time Sample Size   deliveries behind the Material + Vendor pair's own
                          record (0 = the estimate is entirely the vendor's or
                          the overall rate)
    """
    df = df.copy()

    keys = pd.MultiIndex.from_frame(df[[material_col, vendor_col]])

    pair_rate = table["pair"]["late_rate"].reindex(keys).to_numpy(dtype=float)
    pair_n = table["pair"]["n"].reindex(keys).to_numpy(dtype=float)
    vendor_rate = df[vendor_col].map(table["vendor"]["late_rate"]).to_numpy(dtype=float)

    late_rate = np.where(
        ~np.isnan(pair_rate),
        pair_rate,
        np.where(~np.isnan(vendor_rate), vendor_rate, table["global_rate"])
    )

    df["On-Time Probability"] = 1 - late_rate
    df["On-Time Sample Size"] = np.nan_to_num(pair_n, nan=0).astype(int)

    return df