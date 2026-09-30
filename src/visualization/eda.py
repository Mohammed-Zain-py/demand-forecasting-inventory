import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    df = pd.read_csv("data/processed/store_sales_processed.csv", parse_dates=["date"])
    items = pd.read_csv("data/raw/items.csv")
    df = df.merge(items, on='item_nbr', how='left')
    
    os.makedirs("eda_outputs", exist_ok=True)
    
    # 1. Overall demand over time
    plt.figure(figsize=(15, 5))
    daily_demand = df.groupby('date')['unit_sales'].sum()
    daily_demand.plot()
    plt.title('Overall Store 25 Demand Over Time (23 items)')
    plt.ylabel('Total Unit Sales')
    plt.grid(True, alpha=0.3)
    plt.savefig('eda_outputs/01_overall_demand.png')
    plt.close()
    
    # 2. Weekly seasonality
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    plt.figure(figsize=(10, 5))
    weekly = df.groupby('day_name')['unit_sales'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    weekly.plot(kind='bar')
    plt.title('Average Daily Demand by Day of Week')
    plt.ylabel('Average Unit Sales')
    plt.savefig('eda_outputs/02_weekly_seasonality.png')
    plt.close()
    
    # 3. Monthly/Long-term patterns
    df['month'] = df['date'].dt.month
    plt.figure(figsize=(10, 5))
    df.groupby('month')['unit_sales'].mean().plot(kind='bar')
    plt.title('Average Demand by Month')
    plt.ylabel('Average Unit Sales')
    plt.savefig('eda_outputs/03_monthly_seasonality.png')
    plt.close()
    
    # 4. Product demand distributions
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='family', y='unit_sales', data=df)
    plt.xticks(rotation=45)
    plt.title('Demand Distribution by Product Family')
    # Filter out extreme outliers for better visualization
    plt.ylim(-5, df['unit_sales'].quantile(0.99) * 1.5)
    plt.tight_layout()
    plt.savefig('eda_outputs/04_demand_by_family.png')
    plt.close()
    
    # 5. Promotion vs Non-promotion demand
    plt.figure(figsize=(8, 5))
    promo_stats = df.groupby('onpromotion')['unit_sales'].mean()
    promo_stats.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Average Sales: Promotion vs No Promotion')
    plt.ylabel('Average Unit Sales')
    plt.xticks(ticks=[0, 1], labels=['No Promo', 'Promo'], rotation=0)
    plt.savefig('eda_outputs/05_promo_effect.png')
    plt.close()
    
    # 6. Identifying difficult to forecast products
    # Products with high Coefficient of Variation (CV) are harder to forecast
    stats = df.groupby('item_nbr')['unit_sales'].agg(['mean', 'std'])
    stats['cv'] = stats['std'] / stats['mean']
    stats = stats.sort_values('cv', ascending=False)
    stats.to_csv('eda_outputs/product_forecastability.csv')
    
    print("EDA outputs generated in eda_outputs/")

if __name__ == "__main__":
    run_eda()
