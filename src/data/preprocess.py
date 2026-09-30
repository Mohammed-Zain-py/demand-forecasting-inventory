import pandas as pd
import numpy as np
import os

def load_data():
    df = pd.read_csv("data/raw/store_sales.csv", parse_dates=["date"])
    return df

def preprocess_data(df):
    print("Starting preprocessing...")
    
    # 1. Handle negative sales
    # Decision: The dataset contains exactly 1 negative unit_sales observation (-3.0),
    # which represents a product return. Our goal is demand forecasting, and consumer 
    # demand cannot be negative. While a return ideally subtracts from a past sale, we 
    # don't know the origin date. Capping at 0.0 is the most defensible proxy for 
    # "zero new demand" on this specific day.
    neg_count = (df['unit_sales'] < 0).sum()
    print(f"Capping {neg_count} negative unit_sales records to 0.0")
    df['unit_sales'] = df['unit_sales'].clip(lower=0.0)
    
    # 2. Handle missing onpromotion
    # Decision: Promotion tracking was unavailable prior to April 2014.
    # We must distinguish between "known not promoted" and "unknown/not recorded".
    # We add a 'promotion_known' indicator.
    nan_promo_count = df['onpromotion'].isnull().sum()
    print(f"Flagging {nan_promo_count} missing onpromotion records as unknown.")
    df['promotion_known'] = df['onpromotion'].notnull().astype(int)
    # Now we fill the missing with False, but the model can differentiate using promotion_known
    df['onpromotion'] = df['onpromotion'].fillna(False).astype(bool)
    
    # 3. Handle missing dates (Gaps in time series)
    print("Reconstructing continuous time series grid...")
    min_date = df['date'].min()
    max_date = df['date'].max()
    full_dates = pd.date_range(start=min_date, end=max_date) # 1688 days
    
    # Identify globally missing dates (store closed)
    # Verification confirmed these 70 dates had 0 sales across all products in Store 25 globally.
    actual_dates = pd.to_datetime(df['date'].unique())
    global_missing = full_dates.difference(actual_dates)
    print(f"Identified {len(global_missing)} confirmed store closures.")
    
    # Create the grid
    items = df['item_nbr'].unique()
    grid = pd.MultiIndex.from_product(
        [full_dates, [25], items], 
        names=['date', 'store_nbr', 'item_nbr']
    ).to_frame(index=False)
    
    # Merge with actual data
    df_processed = pd.merge(grid, df, on=['date', 'store_nbr', 'item_nbr'], how='left')
    
    # 4. Fill unit_sales = 0 for gaps
    # Of the 2,759 missing combinations:
    # 1,610 are due to store closures.
    # 1,149 are true implicit zeroes (item didn't sell on an active day).
    missing_sales_count = df_processed['unit_sales'].isnull().sum()
    print(f"Filled {missing_sales_count} total missing date-item combinations with 0.0 unit_sales.")
    df_processed['unit_sales'] = df_processed['unit_sales'].fillna(0.0)
    
    # 5. Fill onpromotion and promotion_known for gaps
    df_processed['promotion_known'] = df_processed['promotion_known'].fillna(0).astype(int)
    df_processed['onpromotion'] = df_processed['onpromotion'].fillna(False).astype(bool)
    
    # 6. Add store_closed feature
    # This explicit feature informs the model that demand observability was 0 due to closure.
    df_processed['store_closed'] = df_processed['date'].isin(global_missing).astype(int)
    
    return df_processed

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df_raw = load_data()
    df_processed = preprocess_data(df_raw)
    
    output_path = "data/processed/store_sales_processed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path} with shape {df_processed.shape}")
