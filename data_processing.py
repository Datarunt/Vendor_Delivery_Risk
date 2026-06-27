import pandas as pd

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
        'Name 1', 'Material', 'Date Rcvd', 'Qty Due', 'Qty Rcvd',
        'Description', 'Days Late'
    ]
    
    df = df[KEEP_COLUMNS].copy()
    df.rename(columns={'Name 1': 'Vendor Name'}, inplace=True)

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
    df['Date Rcvd'] = pd.to_datetime(df['Date Rcvd'], errors='coerce')
    df['Days Late'] = pd.to_numeric(df['Days Late'], errors='coerce')

    # ---------------------------
    # 5. ADJUST DAYS LATE FOR OVER-DELIVERY
    df['Days Late'] = df.apply(
        lambda row: 0 if row['Qty Rcvd'] >= row['Qty Due'] else row['Days Late'],
        axis=1
    )
    df = df.dropna(subset=['Days Late', 'Date Rcvd'])

    # ---------------------------
    # 6. AVG DAYS LATE PER VENDOR
    df['Avg Days Late'] = df.groupby('Vendor Name')['Days Late'].transform('mean')

    # ---------------------------
    # 7. ENSURE LATEST VENDOR
    df = df.sort_values(['Material', 'Date Rcvd'])


    return df