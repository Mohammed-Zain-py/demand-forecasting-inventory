import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.models.forecast_model import ForecastModel
from src.evaluation.validation import get_validation_folds

def generate_cv_predictions():
    df_features = pd.read_csv("data/processed/store_sales_features.csv")
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    folds = get_validation_folds()
    
    all_preds = []
    
    for fold_idx, (t_start, t_end, v_start, v_end) in enumerate(folds):
        train_df = df_features[df_features['date'] <= t_end].copy()
        val_df = df_features[(df_features['date'] >= v_start) & (df_features['date'] <= v_end)].copy()
        
        ml_model = ForecastModel()
        ml_model.train(train_df)
        
        val_df['predicted'] = ml_model.predict(val_df)
        val_df['fold'] = fold_idx + 1
        
        all_preds.append(val_df[['date', 'item_nbr', 'unit_sales', 'predicted', 'fold']])
        
    cv_preds_df = pd.concat(all_preds, ignore_index=True)
    cv_preds_df['error'] = cv_preds_df['unit_sales'] - cv_preds_df['predicted']
    
    cv_preds_df.to_csv("data/processed/cv_predictions.csv", index=False)
    print("Out-of-sample CV predictions saved to data/processed/cv_predictions.csv")

if __name__ == "__main__":
    generate_cv_predictions()
