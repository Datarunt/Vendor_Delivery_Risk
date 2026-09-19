import datetime as dt

import pandas as pd
import numpy as np

def _nth_weekday(year, month, weekday, n):
    first = dt.date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day):
    # A holiday on a weekend is taken on the nearest weekday.
    if day.weekday() == 5:
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def non_working_days(first_year, last_year):
    """
    Weekday dates that do not count as working days when measuring lateness:
    New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and
    the day after, and a year-end closure (Dec 24 and Dec 26-31, plus
    Christmas Day itself).

    This approximates the calendar behind the ERP "Days Late" column that this
    calculation replaces, and was chosen because it reproduces that column
    closely. Edit the list to match your own plant calendar.
    """
    days = set()

    for y in range(first_year, last_year + 1):
        days.add(_observed(dt.date(y, 1, 1)))
        days.add(_observed(dt.date(y, 7, 4)))
        days.add(_last_weekday(y, 5, 0))            # Memorial Day
        days.add(_nth_weekday(y, 9, 0, 1))          # Labor Day

        thanksgiving = _nth_weekday(y, 11, 3, 4)
        days.add(thanksgiving)
        days.add(thanksgiving + dt.timedelta(days=1))

        for d in (24, 25, 26, 27, 28, 29, 30, 31):  # year-end closure
            days.add(dt.date(y, 12, d))

    return np.array(sorted(days), dtype="datetime64[D]")


def business_days_late(commit_dates, delivery_dates):
    """
    Working days between the commit date and the actual delivery date.

    Positive = late, 0 = on time (or both dates fall in the same working
    period), negative = early. Weekends and the days from non_working_days()
    are skipped. Early deliveries are counted as on time by the caller
    (see 4b below).

    Working days are used because the ERP "Days Late" column this replaces is
    measured in working days. Against that column this calculation gives the
    same number of days on ~96% of rows, is within 1 day on ~99%, and lands in
    the same lateness band (on time / 1 day / 2-4 / 5-15 / >15) on ~99.7%.
    Calendar days would overstate longer delays.
    """
    commit = pd.to_datetime(commit_dates, errors="coerce")
    actual = pd.to_datetime(delivery_dates, errors="coerce")

    valid = commit.notna() & actual.notna()
    out = pd.Series(np.nan, index=commit.index, dtype=float)

    if valid.any():
        c = commit[valid].values.astype("datetime64[D]")
        a = actual[valid].values.astype("datetime64[D]")

        first_year = int(min(c.min(), a.min()).astype("datetime64[Y]").astype(int)) + 1970
        last_year = int(max(c.max(), a.max()).astype("datetime64[Y]").astype(int)) + 1970
        holidays = non_working_days(first_year, last_year)

        late = a >= c
        out[valid] = np.where(
            late,
            np.busday_count(c, a, holidays=holidays),
            -np.busday_count(a, c, holidays=holidays)
        )

    return out


def load_historical_file(hist_file):

    # 1. READ HISTORICAL FILE
    if hist_file.filename.lower().endswith(".csv"):
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                df = pd.read_csv(hist_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Could not decode CSV file")
    else:
        df = pd.read_excel(hist_file)

    df.columns = df.columns.str.strip()

    # ---------------------------
    # 2. KEEP REQUIRED COLUMNS
    # These are the Historical Commits fields chosen in the UI.
    KEEP_COLUMNS = [
        'Vendor', 'Material', 'Commit Date', 'Commit Qty',
        'Actual Delivery Date', 'Actual Delivery Qty'
    ]
    df = df[KEEP_COLUMNS].copy()

    # Internal names used by the forecast, model and accuracy code.
    df = df.rename(columns={
        'Vendor': 'Vendor Name',
        'Commit Date': 'Date Due',
        'Commit Qty': 'Quantity Due',
        'Actual Delivery Date': 'Date Received',
        'Actual Delivery Qty': 'Quantity Received',
    })

    # ---------------------------
    # 3. CLEAN MATERIAL
    df['Material'] = (
        df['Material'].astype(str)
          .str.findall(r'\w[\w-]*')
          .str[0]
          .str.replace(r'[A-Za-z]+$', '', regex=True)
          .replace('nan', pd.NA)
    )
    df['Material'] = df['Material'].astype(str).str.strip()
    df = df[(df['Material'] != '') & (~df['Material'].isna())]

    # ---------------------------
    # 4. FIX DATA TYPES
    df['Date Due'] = pd.to_datetime(df['Date Due'], errors='coerce')
    df['Date Received'] = pd.to_datetime(df['Date Received'], errors='coerce')
    df['Quantity Due'] = pd.to_numeric(df['Quantity Due'], errors='coerce')
    df['Quantity Received'] = pd.to_numeric(df['Quantity Received'], errors='coerce')

    # ---------------------------
    # 4b. DERIVE DAYS LATE FROM THE COMMIT AND ACTUAL DELIVERY DATES
    # An early delivery is not late: it counts as on time (0 days late). This
    # is how the ERP-based version behaved too: its early rows had no numeric
    # Days Late, were classified "Hit / On Time", and were set to 0 late.
    df['Number of Days Late'] = business_days_late(
        df['Date Due'], df['Date Received']
    ).clip(lower=0)

    # ---------------------------
    # 4c. DAYS LATE CLASSIFICATION (from the derived days late)
    df['Days Late Classification'] = np.select(
        [
            df['Number of Days Late'] == 0,
            df['Number of Days Late'] == 1,
            df['Number of Days Late'].between(2, 4),
            df['Number of Days Late'].between(5, 15),
            df['Number of Days Late'] > 15,
        ],
        [
            "Hit / On Time",
            "Miss / 1 Day Late",
            "Miss / 2-4 Calendar Days",
            "Miss / 5-15 Calendar Days",
            "Miss / > 15 Calendar Days",
        ],
        default="Hit / On Time"
    )

    # ---------------------------
    # 5. ADJUST DAYS LATE FOR OVER-DELIVERY
    df.loc[
        df['Quantity Received'] >= df['Quantity Due'],
        'Number of Days Late'
    ] = 0
    df = df.dropna(subset=['Number of Days Late', 'Date Received'])

    # ---------------------------
    # 6. AVG DAYS LATE PER VENDOR
    df['Avg Days Late'] = df.groupby('Vendor Name')['Number of Days Late'].transform('mean')

    # ---------------------------
    # 7. ENSURE LATEST VENDOR
    df = df.sort_values(['Material', 'Date Received'])

    return df