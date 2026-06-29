import pandas as pd
import numpy as np

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
    KEEP_COLUMNS = [
        'Vendor', 'Material', 'Date Received', 'Quantity Due', 'Quantity Received',
        'Number of Days Late'
    ]
    df = df[KEEP_COLUMNS].copy()

    # add Days Late Classification if not present
    if 'Days Late Classification' not in df.columns:
        df['Days Late Classification'] = None
    df.rename(columns={'Vendor': 'Vendor Name'}, inplace=True)

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
    df['Date Received'] = pd.to_datetime(df['Date Received'], errors='coerce')
    df['Number of Days Late'] = pd.to_numeric(df['Number of Days Late'], errors='coerce')

    # ---------------------------
    # 4b. DERIVE DAYS LATE CLASSIFICATION IF MISSING
    if 'Days Late Classification' not in df.columns or df['Days Late Classification'].isna().all():
        df['Days Late Classification'] = np.select(
            [
                df['Number of Days Late'] < 0,
                df['Number of Days Late'] == 0,
                df['Number of Days Late'] == 1,
                df['Number of Days Late'].between(2, 4),
                df['Number of Days Late'].between(5, 15),
                df['Number of Days Late'] > 15,
            ],
            [
                "Miss / Early",
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
    df['Number of Days Late'] = df.apply(
        lambda row: 0 if row['Quantity Received'] >= row['Quantity Due'] else row['Number of Days Late'],
        axis=1
    )
    df = df.dropna(subset=['Number of Days Late', 'Date Received'])

    # ---------------------------
    # 6. AVG DAYS LATE PER VENDOR
    df['Avg Days Late'] = df.groupby('Vendor Name')['Number of Days Late'].transform('mean')

    # ---------------------------
    # 7. ENSURE LATEST VENDOR
    df = df.sort_values(['Material', 'Date Received'])

    return df