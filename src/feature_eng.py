import numpy as np
import pandas as pd

def create_features(df):
    """
    Creates time-series features for DAM price forecasting.

    Parameters:
        df (pd.DataFrame): Preprocessed dataframe with 'datetime' column

    Returns:
        df (pd.DataFrame): Dataframe with features
        feature_cols (list): List of feature column names
        target_col (str): Target column name
    """

    df = df.copy()

    # ==============================
    # 0. Basic time features
    # ==============================
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    # ==============================
    # 1. Cyclical temporal features
    # ==============================
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']   = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # ==============================
    # 2. Calendar flags
    # ==============================
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_sunday']  = (df['dayofweek'] == 6).astype(int)
    df['is_friday']  = (df['dayofweek'] == 4).astype(int)
    df['is_midweek'] = df['dayofweek'].isin([2, 3]).astype(int)

    # ==============================
    # 3. Interaction features
    # ==============================
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']

    df['is_weekday_evenpeak'] = (
        (df['hour'].between(15, 22)) & (df['is_weekend'] == 0)
    ).astype(int)

    df['is_weekday_morningpeak'] = (
        (df['hour'].between(6, 9)) & (df['is_weekend'] == 0)
    ).astype(int)

    df['is_sunday_night'] = (
        (df['hour'].between(1, 4)) & (df['is_sunday'] == 1)
    ).astype(int)

    # ==============================
    # 4. Price lag features
    # ==============================
    df['lag_1']   = df['Price (Rs./MWh)'].shift(1)
    df['lag_4']   = df['Price (Rs./MWh)'].shift(4)
    df['lag_96']  = df['Price (Rs./MWh)'].shift(96)
    df['lag_672'] = df['Price (Rs./MWh)'].shift(672)

    # ==============================
    # 5. Rolling statistics
    # ==============================
    df['roll_mean_4']  = df['Price (Rs./MWh)'].rolling(4).mean()
    df['roll_mean_96'] = df['Price (Rs./MWh)'].rolling(96).mean()
    df['roll_std_96']  = df['Price (Rs./MWh)'].rolling(96).std()
    df['roll_max_96']  = df['Price (Rs./MWh)'].rolling(96).max()

    # ==============================
    # 6. Supply-side features
    # ==============================
    df['cleared_sell_lag1']  = df['Cleared Sell (MW)'].shift(1)
    df['cleared_sell_lag96'] = df['Cleared Sell (MW)'].shift(96)
    df['sell_roll_mean_4']   = df['Cleared Sell (MW)'].rolling(4).mean()
    df['sell_roll_mean_96']  = df['Cleared Sell (MW)'].rolling(96).mean()

    # ==============================
    # 7. Imbalance (regime split)
    # ==============================
    df['imbalance'] = df['Cleared Buy (MW)'] - df['Cleared Sell (MW)']

    df['imbalance_offpeak'] = df['imbalance'].where(df['hour'] <= 13, 0)
    df['imbalance_peak']    = df['imbalance'].where(df['hour'].between(14, 16), 0)
    df['imbalance_lag96']   = df['imbalance'].shift(96)

    # ==============================
    # 8. Anomaly flag
    # ==============================
    df['is_zero_buy'] = (df['Cleared Buy (MW)'] == 0).astype(int)

    # ==============================
    # 9. Drop NaNs (due to lag)
    # ==============================
    before = len(df)
    df = df.dropna(subset=['lag_672'])
    after = len(df)

    print(f"Dropped {before - after} rows due to lag features")

    # ==============================
    # 10. Feature list
    # ==============================
    feature_cols = [
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos',

        'is_weekend', 'is_sunday', 'is_friday', 'is_midweek',

        'hour_x_weekend',
        'is_weekday_evenpeak',
        'is_weekday_morningpeak',
        'is_sunday_night',

        'lag_1', 'lag_4', 'lag_96', 'lag_672',

        'roll_mean_4', 'roll_mean_96', 'roll_std_96', 'roll_max_96',

        'Cleared Sell (MW)',
        'cleared_sell_lag1', 'cleared_sell_lag96',
        'sell_roll_mean_4', 'sell_roll_mean_96',

        'imbalance_offpeak', 'imbalance_peak', 'imbalance_lag96',

        'is_zero_buy',
    ]

    target_col = 'Price (Rs./MWh)'

    # ==============================
    # 11. Summary logs
    # ==============================
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Training samples: {len(df)}")

    return df, feature_cols, target_col
