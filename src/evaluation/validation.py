import pandas as pd
import os

def get_validation_folds():
    """
    Returns a list of tuples containing (train_start, train_end, val_start, val_end).
    We use rolling-origin validation.
    The final test set is 2017-08-01 to 2017-08-15.
    We will create 6 validation folds prior to this test set, each 30 days long.
    """
    test_start = pd.to_datetime('2017-08-01')
    folds = []
    
    # Generate 6 folds of 30 days, ending right before the test start
    for i in range(6, 0, -1):
        val_end = test_start - pd.Timedelta(days=1 + (i - 1) * 30)
        val_start = val_end - pd.Timedelta(days=29)
        train_end = val_start - pd.Timedelta(days=1)
        
        folds.append((
            '2013-01-01',
            train_end.strftime('%Y-%m-%d'),
            val_start.strftime('%Y-%m-%d'),
            val_end.strftime('%Y-%m-%d')
        ))
        
    return folds

def get_test_period():
    return ('2017-08-01', '2017-08-15')
