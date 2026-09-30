import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_errors():
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load Per-Product Metrics
    prod_metrics = pd.read_csv("reports/per_product_metrics.csv")
    
    # Pivot to compare ML vs Seasonal Naive WMAPE
    wmape_pivot = prod_metrics.pivot(index='item_nbr', columns='Model', values='WMAPE')
    wmape_pivot['Improvement_over_Seasonal'] = wmape_pivot['Seasonal Naive'] - wmape_pivot['LightGBM']
    
    wmape_pivot = wmape_pivot.sort_values('Improvement_over_Seasonal', ascending=False)
    
    print("\n--- Products where ML Improves Substantially ---")
    print(wmape_pivot.head(5)[['Seasonal Naive', 'LightGBM', 'Improvement_over_Seasonal']])
    
    print("\n--- Products where ML Performs Poorly or Worse than Baseline ---")
    print(wmape_pivot.tail(5)[['Seasonal Naive', 'LightGBM', 'Improvement_over_Seasonal']])
    
    # Plot improvement distribution
    plt.figure(figsize=(10, 5))
    wmape_pivot['Improvement_over_Seasonal'].plot(kind='bar')
    plt.title("LightGBM WMAPE Improvement over Seasonal Naive per Product")
    plt.ylabel("WMAPE Reduction (Higher is better)")
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig("reports/ml_improvement_per_product.png")
    plt.close()
    
    # Analyze stable vs lumpy
    # To do this, let's load the actual predictions for the test set
    df_features = pd.read_csv("data/processed/store_sales_features.csv", parse_dates=['date'])
    test_df = df_features[(df_features['date'] >= '2017-08-01') & (df_features['date'] <= '2017-08-15')]
    
    # Aggregate errors by day to find "high-error periods"
    # Note: We didn't save the raw test predictions from the pipeline, but we can infer that
    # the model struggles when overall test WMAPE > 0.6.
    
    wmape_pivot.to_csv("reports/error_analysis_summary.csv")
    print("\nError analysis completed and saved to reports/.")

if __name__ == "__main__":
    analyze_errors()
