import pandas as pd
import numpy as np
import os

def create_features(df_sales, df_items):
    print("Starting feature engineering...")
    
    # Merge items metadata
    df = pd.merge(df_sales, df_items, on='item_nbr', how='left')
    
    # 1. Calendar Features
    print("Creating calendar features...")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Sort strictly by item and date for time-series features
    df = df.sort_values(by=['item_nbr', 'date']).reset_index(drop=True)
    
    # 2. Lag Features & Rolling Features (Leakage-Safe)
    print("Creating leakage-safe lag and rolling features...")
    
    # To prevent leakage, we first shift the target (unit_sales) by 1 day.
    # This represents the "latest available observation" for prediction on day D.
    df['shifted_sales_1'] = df.groupby('item_nbr')['unit_sales'].shift(1)
    
    # Lags based on the target directly
    df['lag_1'] = df['shifted_sales_1']
    df['lag_7'] = df.groupby('item_nbr')['unit_sales'].shift(7)
    df['lag_14'] = df.groupby('item_nbr')['unit_sales'].shift(14)
    df['lag_28'] = df.groupby('item_nbr')['unit_sales'].shift(28)
    
    # Rolling features computed strictly on the 1-day shifted sales
    # so that rolling_mean_7 for day D covers D-7 to D-1.
    df['rolling_mean_7'] = df.groupby('item_nbr')['shifted_sales_1'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_mean_14'] = df.groupby('item_nbr')['shifted_sales_1'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean())
    df['rolling_mean_28'] = df.groupby('item_nbr')['shifted_sales_1'].transform(
        lambda x: x.rolling(window=28, min_periods=1).mean())
        
    df['rolling_std_7'] = df.groupby('item_nbr')['shifted_sales_1'].transform(
        lambda x: x.rolling(window=7, min_periods=2).std())
    df['rolling_std_28'] = df.groupby('item_nbr')['shifted_sales_1'].transform(
        lambda x: x.rolling(window=28, min_periods=2).std())
        
    # Fill standard deviation NaNs (which happen if rolling window has < 2 obs) with 0
    df['rolling_std_7'] = df['rolling_std_7'].fillna(0.0)
    df['rolling_std_28'] = df['rolling_std_28'].fillna(0.0)
    
    # Drop intermediate shifted column
    df = df.drop(columns=['shifted_sales_1'])
    
    # 3. Handle early rows with missing lags
    # Because lag_28 requires 28 days of history, the first 28 days for each item
    # will have NaN in lag_28.
    # Treatment: We explicitly drop the first 28 days of data for all items 
    # rather than filling with arbitrary zeros, ensuring the model only learns 
    # from complete feature sets.
    initial_shape = df.shape
    df = df.dropna(subset=['lag_28']).reset_index(drop=True)
    final_shape = df.shape
    print(f"Dropped early rows due to missing lag_28: {initial_shape[0]} -> {final_shape[0]}")
    
    return df

if __name__ == "__main__":
    os.makedirs("src/features", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    df_sales = pd.read_csv("data/processed/store_sales_processed.csv")
    df_items = pd.read_csv("data/raw/items.csv")
    
    df_features = create_features(df_sales, df_items)
    df_features.to_csv("data/processed/store_sales_features.csv", index=False)
    print("Features saved to data/processed/store_sales_features.csv")
