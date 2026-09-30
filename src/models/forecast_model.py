import pandas as pd
import numpy as np
import lightgbm as lgb

class ForecastModel:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        )
        self.features = [
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_28',
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'is_weekend',
            'onpromotion', 'promotion_known', 'store_closed',
            'item_nbr' # Use item_nbr as a categorical feature
        ]
        self.categorical_features = ['item_nbr', 'day_of_week', 'month']
        
    def train(self, df_train):
        X_train = df_train[self.features].copy()
        y_train = df_train['unit_sales']
        
        # Convert categorical to category dtype for LightGBM
        for cat in self.categorical_features:
            X_train[cat] = X_train[cat].astype('category')
            
        self.model.fit(
            X_train, y_train,
            categorical_feature=self.categorical_features
        )
        
    def predict(self, df_test):
        X_test = df_test[self.features].copy()
        for cat in self.categorical_features:
            X_test[cat] = X_test[cat].astype('category')
            
        preds = self.model.predict(X_test)
        # Cap predictions at 0 because demand cannot be negative
        return np.clip(preds, 0, None)
