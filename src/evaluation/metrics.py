import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def wmape(y_true, y_pred):
    """
    Weighted Mean Absolute Percentage Error.
    Formula: sum(|y_true - y_pred|) / sum(|y_true|)
    This is highly preferred in retail over MAPE because it avoids 
    division by zero on days where actual demand is 0.
    """
    sum_true = np.sum(np.abs(y_true))
    if sum_true == 0:
        return 0.0 # Perfect prediction or undefined
    return np.sum(np.abs(y_true - y_pred)) / sum_true

def evaluate_predictions(y_true, y_pred):
    """Returns a dictionary of metrics."""
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'WMAPE': wmape(y_true, y_pred)
    }
