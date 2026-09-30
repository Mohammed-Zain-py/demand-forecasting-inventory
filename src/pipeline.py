import pandas as pd
import numpy as np
import os
import sys

# Ensure relative imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from features.feature_engineering import create_features
from evaluation.validation import get_validation_folds, get_test_period
from evaluation.metrics import evaluate_predictions
from models.baseline import BaselineModels
from models.forecast_model import ForecastModel
from models.lstm_forecast import PyTorchLSTMModel

def main():
    print("=== DEMAND FORECASTING PIPELINE ===")
    
    # 1. Feature Engineering
    if not os.path.exists("data/processed/store_sales_features.csv"):
        df_sales = pd.read_csv("data/processed/store_sales_processed.csv")
        df_items = pd.read_csv("data/raw/items.csv")
        df_features = create_features(df_sales, df_items)
        df_features.to_csv("data/processed/store_sales_features.csv", index=False)
    else:
        df_features = pd.read_csv("data/processed/store_sales_features.csv")
    
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    folds = get_validation_folds()
    test_start, test_end = get_test_period()
    
    models_to_evaluate = ['Naive', 'Seasonal Naive', 'LightGBM', 'PyTorch LSTM']
    overall_results = []
    product_results = []
    
    baseline = BaselineModels()
    
    print("\n=== RUNNING VALIDATION FOLDS ===")
    for fold_idx, (t_start, t_end, v_start, v_end) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}: Train <= {t_end} | Val = {v_start} to {v_end}")
        
        train_df = df_features[df_features['date'] <= t_end].copy()
        val_df = df_features[(df_features['date'] >= v_start) & (df_features['date'] <= v_end)].copy()
        
        # Train ML Model
        ml_model = ForecastModel()
        ml_model.train(train_df)
        
        # Train LSTM Model
        lstm_model = PyTorchLSTMModel()
        lstm_model.train(train_df)
        
        # Predict
        val_df['Naive_pred'] = baseline.naive_predict(val_df)
        val_df['Seasonal Naive_pred'] = baseline.seasonal_naive_predict(val_df)
        val_df['LightGBM_pred'] = ml_model.predict(val_df)
        val_df['PyTorch LSTM_pred'] = lstm_model.predict(val_df)
        
        y_true = val_df['unit_sales'].values
        
        for m in models_to_evaluate:
            y_pred = val_df[f'{m}_pred'].values
            metrics = evaluate_predictions(y_true, y_pred)
            metrics['Fold'] = fold_idx + 1
            metrics['Model'] = m
            overall_results.append(metrics)
            
    print("\n=== RUNNING TEST SET ===")
    # Train on everything before test
    train_df = df_features[df_features['date'] < test_start].copy()
    test_df = df_features[(df_features['date'] >= test_start) & (df_features['date'] <= test_end)].copy()
    
    ml_model = ForecastModel()
    ml_model.train(train_df)
    
    lstm_model = PyTorchLSTMModel()
    lstm_model.train(train_df)
    
    test_df['Naive_pred'] = baseline.naive_predict(test_df)
    test_df['Seasonal Naive_pred'] = baseline.seasonal_naive_predict(test_df)
    test_df['LightGBM_pred'] = ml_model.predict(test_df)
    test_df['PyTorch LSTM_pred'] = lstm_model.predict(test_df)
    
    y_true_test = test_df['unit_sales'].values
    
    test_metrics = []
    for m in models_to_evaluate:
        y_pred = test_df[f'{m}_pred'].values
        metrics = evaluate_predictions(y_true_test, y_pred)
        metrics['Fold'] = 'Test'
        metrics['Model'] = m
        test_metrics.append(metrics)
        
    overall_results.extend(test_metrics)
    
    # Save overall metrics
    results_df = pd.DataFrame(overall_results)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/overall_metrics.csv", index=False)
    
    print("\n--- Validation Folds Summary (Across Folds) ---")
    val_res = results_df[results_df['Fold'] != 'Test']
    summary = val_res.groupby('Model')['WMAPE'].agg(['mean', 'median', 'std']).reset_index()
    print(summary)
    summary.to_csv("reports/fold_summary_metrics.csv", index=False)
    
    print("\n--- Overall Metrics ---")
    print(results_df[['Model', 'Fold', 'MAE', 'RMSE', 'WMAPE']])
    
    # Per-product results on TEST SET
    print("\n--- Per-Product Analysis on TEST SET ---")
    for item in test_df['item_nbr'].unique():
        item_df = test_df[test_df['item_nbr'] == item]
        y_true_item = item_df['unit_sales'].values
        
        for m in models_to_evaluate:
            y_pred_item = item_df[f'{m}_pred'].values
            metrics = evaluate_predictions(y_true_item, y_pred_item)
            metrics['item_nbr'] = item
            metrics['Model'] = m
            product_results.append(metrics)
            
    prod_results_df = pd.DataFrame(product_results)
    prod_results_df.to_csv("reports/per_product_metrics.csv", index=False)
    
    # Compare models on Test set
    print("\nTest Set WMAPE by Model:")
    test_res = results_df[results_df['Fold'] == 'Test']
    for _, row in test_res.iterrows():
        print(f"{row['Model']}: {row['WMAPE']:.4f}")
        
    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
