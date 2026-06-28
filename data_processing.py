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
        'Vendor', 'Material', 'Date Received', 'Quantity Due', 'Quantity Received',
        'Days Late Classification', 'Number of Days Late'
    ]
    df = df[KEEP_COLUMNS].copy()
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