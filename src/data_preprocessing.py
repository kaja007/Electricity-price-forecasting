import pandas as pd
import glob

def load_and_preprocess_data(folder_path):
    """
    Loads multiple CSV files, merges them, checks missing values,
    and creates a datetime column.

    Parameters:
        folder_path (str): Path to folder containing CSV files (e.g., "data/")

    Returns:
        pd.DataFrame: Cleaned and processed dataframe
    """

    file_path = f"{folder_path}/*.csv"
    files = glob.glob(file_path)

    if len(files) == 0:
        raise ValueError("No CSV files found in the given folder")

    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)


    df_april_oct = pd.concat(df_list, ignore_index=True)

    
    na_summary = df_april_oct.isna().sum()
    na_percent = (df_april_oct.isna().mean() * 100).round(2)

    print("\nMissing Values Count:\n", na_summary)
    print("\nMissing Values %:\n", na_percent)

    
    df_april_oct['start_time'] = df_april_oct['Time Period'].str.split('-').str[0]

    df_april_oct['datetime'] = pd.to_datetime(
        df_april_oct['Delivery Date'] + ' ' + df_april_oct['start_time'],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )

    before = df_april_oct.shape[0]
    df_april_oct = df_april_oct.dropna(subset=['datetime'])
    after = df_april_oct.shape[0]

    print(f"\nDropped {before - after} rows due to invalid datetime")

    df_april_oct = df_april_oct.sort_values('datetime').reset_index(drop=True)

    return df_april_oct
