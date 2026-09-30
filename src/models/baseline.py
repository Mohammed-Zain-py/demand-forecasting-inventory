import pandas as pd
import numpy as np

class BaselineModels:
    def __init__(self):
        pass
        
    def naive_predict(self, df):
        """Predicts today's demand using exactly yesterday's demand (lag_1)"""
        return df['lag_1']
        
    def seasonal_naive_predict(self, df):
        """Predicts today's demand using the same day last week (lag_7)"""
        return df['lag_7']
