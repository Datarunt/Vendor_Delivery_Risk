import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from lateness_services import LATE_AFTER_WORKING_DAYS, PRIOR_STRENGTH

# ---------------------------------------------------------------------------
# MACHINE-LEARNED MISS MODEL
# ---------------------------------------------------------------------------
# Every past commit in Historical Commits is one training example:
#
#   outcome   missed (1) if the delivery arrived late (LATE_AFTER_WORKING_DAYS or
#             more working days after the commit date) OR short (less than the
#             committed quantity); otherwise on time and in full (0)
#   inputs    what was knowable when the commit was still open: the vendor's and
#             the material's delivery record up to that point, how big the commit
#             is compared with what this material usually is, and the time of year
#
# The model learns how those combine (for example: a vendor that is fine on
# small commits but slips on large ones, or one that has been getting worse
# lately), which the plain late-rate scorecard in lateness_services.py cannot.
#
# NO PEEKING: inputs for a commit use only deliveries received BEFORE that
# commit's "as-of" date, never the commit's own outcome or anything after it.
# For training, as-of is the commit date minus lead_days, so the inputs are as
# stale as they will be when the model is used on the commits in the forecast
# window (see choose_lead_days).

# Look-back for the "recent" features, in days
RECENT_DAYS = 180

# "Severely late" for the severe-late-rate feature, in working days
SEVERE_LATE_DAYS = 5

# Used when the staleness cannot be measured (no commits in the window)
DEFAULT_LEAD_DAYS = 30

# Below this many training examples the model is skipped
MIN_TRAINING_ROWS = 500

MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,   # a split needs real support: history is noisy
    reg_lambda=5.0,
    random_state=42,
    eval_metric="logloss",
)

FEATURES = [
    # the commit itself
    "commit_qty",
    "qty_vs_pair_avg",        # commit size / this material's usual commit size
    "month",
    "day_of_week",
    # the vendor's record
    "vendor_late_rate",
    "vendor_avg_days_late",
    "vendor_severe_rate",
    "vendor_late_rate_recent",
    # this vendor on this material
    "pair_late_rate",
    "pair_avg_days_late",
    "pair_late_rate_recent",
    # the plain scorecard estimate (lateness_services.py), so the model starts
    # from it and only has to learn what it misses
    "scorecard_late_rate",
]

# Not inputs on purpose: how MANY deliveries a vendor or material has on record
# (and how many in the recent window). Counts grow with calendar time, so the
# model used them as a stand-in for "which era is this" and learned a pattern
# that reversed on newer commits. With them, the model scored worse than
# chance on the saved evaluation runs; without them it matched the scorecard.
# The scorecard estimate still accounts for thin records (it blends them toward
# broader rates).

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days(dates):
    """Datetimes -> whole days since 1970-01-01 (float array, NaN for missing)."""
    d = pd.to_datetime(dates, errors="coerce")
    out = d.values.astype("datetime64[D]").astype("int64").astype(float)
    out[d.isna().to_numpy()] = np.nan
    return out


def _prepare_events(history):
    """One row per past delivery, with the flags the features and outcome need."""
    ev = history[[
        "Material", "Vendor Name", "Date Due", "Date Received",
        "Quantity Due", "Quantity Received", "Working Days Late"
    ]].copy()

    ev["Date Due"] = pd.to_datetime(ev["Date Due"], errors="coerce")
    ev["Date Received"] = pd.to_datetime(ev["Date Received"], errors="coerce")

    for c in ("Quantity Due", "Quantity Received", "Working Days Late"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")

    ev = ev.dropna(subset=["Date Received", "Working Days Late"]).copy()

    wdl = ev["Working Days Late"]
    ev["late"] = (wdl >= LATE_AFTER_WORKING_DAYS).astype(float)
    ev["severe"] = (wdl >= SEVERE_LATE_DAYS).astype(float)
    ev["wdl_pos"] = wdl.clip(lower=0)

    ev["qty_due"] = ev["Quantity Due"].fillna(0.0)
    ev["qty_cnt"] = ev["Quantity Due"].notna().astype(float)

    short = (ev["Quantity Received"] < ev["Quantity Due"]).fillna(False)
    ev["miss"] = ((ev["late"] == 1) | short).astype(int)

    ev["_day"] = _days(ev["Date Received"]).astype("int64")

    return ev.sort_values("_day")


def _sums(events, key, queries, asof_days):
    """
    For each query row: totals over the events with the same key that were
    received STRICTLY BEFORE the query's as-of date (all time, and the last
    RECENT_DAYS).
    """
    n_q = len(queries)
    names = ["n", "late", "wdl", "severe", "qty", "qty_cnt", "n_recent", "late_recent"]
    out = {k: np.zeros(n_q) for k in names}

    groups = {k: g for k, g in events.groupby(key)}

    for k, pos in queries.groupby(key).indices.items():
        g = groups.get(k)

        if g is None:
            continue

        d = g["_day"].to_numpy()

        cs = {
            c: np.concatenate([[0.0], np.cumsum(g[c].to_numpy(dtype=float))])
            for c in ("late", "wdl_pos", "severe", "qty_due", "qty_cnt")
        }

        a = asof_days[pos]

        # deliveries received before the as-of date
        hi = np.searchsorted(d, a, side="left")
        lo = np.searchsorted(d, a - RECENT_DAYS, side="left")

        out["n"][pos] = hi
        out["late"][pos] = cs["late"][hi]
        out["wdl"][pos] = cs["wdl_pos"][hi]
        out["severe"][pos] = cs["severe"][hi]
        out["qty"][pos] = cs["qty_due"][hi]
        out["qty_cnt"][pos] = cs["qty_cnt"][hi]
        out["n_recent"][pos] = hi - lo
        out["late_recent"][pos] = cs["late"][hi] - cs["late"][lo]

    return out


def _rate(num, den):
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def compute_features(events, queries):
    """
    Model inputs for each query row.

    queries needs Material, Vendor Name, as_of, commit_date and commit_qty.
    Only events received before the row's as_of date are used.
    """
    q = queries.reset_index(drop=True)
    asof_days = _days(q["as_of"])

    v = _sums(events, "Vendor Name", q, asof_days)
    p = _sums(events, ["Material", "Vendor Name"], q, asof_days)
    g = _sums(events.assign(_all=1), "_all", q.assign(_all=1), asof_days)

    g_rate = _rate(g["late"], g["n"])

    # the plain scorecard (same blend as lateness_services.py), as of each date
    k = PRIOR_STRENGTH
    vendor_shrunk = (v["late"] + k * g_rate) / (v["n"] + k)
    pair_shrunk = (p["late"] + k * vendor_shrunk) / (p["n"] + k)

    commit_qty = pd.to_numeric(q["commit_qty"], errors="coerce").to_numpy(dtype=float)
    pair_avg_qty = _rate(p["qty"], p["qty_cnt"])
    qty_vs_avg = np.where(pair_avg_qty > 0, commit_qty / pair_avg_qty, np.nan)

    commit_date = pd.to_datetime(q["commit_date"], errors="coerce")

    feats = pd.DataFrame({
        "commit_qty": commit_qty,
        "qty_vs_pair_avg": qty_vs_avg,
        "month": commit_date.dt.month.astype(float),
        "day_of_week": commit_date.dt.dayofweek.astype(float),
        "vendor_n": v["n"],
        "vendor_late_rate": _rate(v["late"], v["n"]),
        "vendor_avg_days_late": _rate(v["wdl"], v["n"]),
        "vendor_severe_rate": _rate(v["severe"], v["n"]),
        "vendor_n_recent": v["n_recent"],
        "vendor_late_rate_recent": _rate(v["late_recent"], v["n_recent"]),
        "pair_n": p["n"],
        "pair_late_rate": _rate(p["late"], p["n"]),
        "pair_avg_days_late": _rate(p["wdl"], p["n"]),
        "pair_n_recent": p["n_recent"],
        "pair_late_rate_recent": _rate(p["late_recent"], p["n_recent"]),
        "scorecard_late_rate": pair_shrunk,
    })

    return feats[FEATURES]


def choose_lead_days(history, commit_dates, cutoff_date):
    """
    How stale the history will be for the commits being scored: the typical gap
    between the newest delivery on record (before the forecast start) and the
    commit dates. Training uses the same gap, so the model is trained on inputs
    that are as out of date as the ones it will be given.
    """
    received = pd.to_datetime(history["Date Received"], errors="coerce")
    received = received[received < pd.to_datetime(cutoff_date)]

    commits = pd.to_datetime(pd.Series(commit_dates), errors="coerce").dropna()

    if received.empty or commits.empty:
        return DEFAULT_LEAD_DAYS

    gap = (commits - received.max()).dt.days.median()

    return int(min(max(gap, 0), 365))


# ---------------------------------------------------------------------------
# Train and predict
# ---------------------------------------------------------------------------

def build_training_set(events, lead_days):
    """
    One row per past commit: its inputs as of (commit date - lead_days), and
    whether it was missed.

    Commits already delivered before their as-of date are left out: they were
    no longer open, so there would have been nothing to predict.
    """
    t = events.dropna(subset=["Date Due"]).copy()

    as_of = t["Date Due"] - pd.Timedelta(days=lead_days)

    queries = pd.DataFrame({
        "Material": t["Material"].to_numpy(),
        "Vendor Name": t["Vendor Name"].to_numpy(),
        "as_of": as_of.to_numpy(),
        "commit_date": t["Date Due"].to_numpy(),
        "commit_qty": t["Quantity Due"].to_numpy(),
    })

    X = compute_features(events, queries)

    still_open = (t["Date Received"].to_numpy() >= as_of.to_numpy())

    y = t["miss"].to_numpy()

    return X[still_open], y[still_open]


def train_miss_model(history, cutoff_date, lead_days=None):
    """
    Trains the miss model on deliveries received before cutoff_date.

    Returns a dict (model, events, cutoff, as_of, lead_days, n_train, miss_rate), or
    None when there is not enough history or the history lacks the columns.
    """
    needed = {"Material", "Vendor Name", "Date Due", "Date Received",
              "Quantity Due", "Quantity Received", "Working Days Late"}

    if not needed.issubset(history.columns):
        print(
            "WARNING: the history is missing columns the miss model needs "
            f"({sorted(needed - set(history.columns))}); it was skipped."
        )
        return None

    cutoff = pd.to_datetime(cutoff_date)

    events = _prepare_events(history)
    events = events[events["Date Received"] < cutoff]

    lead = DEFAULT_LEAD_DAYS if lead_days is None else int(lead_days)

    X, y = build_training_set(events, lead)

    if len(y) < MIN_TRAINING_ROWS or len(np.unique(y)) < 2:
        print(
            f"WARNING: only {len(y)} usable past commits, so the miss model "
            f"was skipped (needs {MIN_TRAINING_ROWS})."
        )
        return None

    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X.to_numpy(dtype=float), y)

    print(
        f"\nMiss model: trained on {len(y)} past commits "
        f"(missed {y.mean():.1%}), inputs as of {lead} days before each commit."
    )

    # The day the model's knowledge of deliveries ends. Usually the forecast
    # start, but when the history stops earlier (the newest delivery on record
    # is months old), the look-back windows must be measured from THAT day. This
    # is the same situation the training examples were built to look like:
    # inputs as of (commit date - lead_days), with nothing newer known.
    as_of = min(cutoff, events["Date Received"].max() + pd.Timedelta(days=1))

    return {
        "model": model,
        "events": events,
        "cutoff": cutoff,
        "as_of": as_of,
        "lead_days": lead,
        "n_train": int(len(y)),
        "miss_rate": float(y.mean()),
    }


def predict_on_time_probability(bundle, commits):
    """
    Chance each commit is met (on time and in full), as a 0-1 array.

    commits needs Material, Vendor Name, Commit Date and Commit Qty (for a
    week or month bucket, the average commit size). Inputs use only deliveries
    received before the forecast start, as of the day the history ends.
    """
    queries = pd.DataFrame({
        "Material": commits["Material"].to_numpy(),
        "Vendor Name": commits["Vendor Name"].to_numpy(),
        "as_of": np.repeat(np.datetime64(bundle["as_of"].to_datetime64()), len(commits)),
        "commit_date": commits["Commit Date"].to_numpy(),
        "commit_qty": commits["Commit Qty"].to_numpy(),
    })

    X = compute_features(bundle["events"], queries)

    p_miss = bundle["model"].predict_proba(X.to_numpy(dtype=float))[:, 1]

    return np.clip(1 - p_miss, 0.001, 0.999)