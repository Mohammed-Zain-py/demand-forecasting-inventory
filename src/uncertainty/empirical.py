import pandas as pd
import numpy as np

class EmpiricalUncertainty:
    def __init__(self, cv_preds_path="data/processed/cv_predictions.csv"):
        self.cv_preds = pd.read_csv(cv_preds_path, parse_dates=['date'])
        self.error_stats = {}
        self._calculate_product_uncertainty()
        
    def _calculate_product_uncertainty(self):
        """
        Calculates out-of-sample error statistics per product.
        error = actual_demand - predicted_demand
        Positive error = under-forecast (demand exceeded forecast) -> risk of stockout.
        Negative error = over-forecast -> risk of excess inventory.
        """
        df = self.cv_preds.copy()
        # Sort chronologically to ensure rolling sums make sense if needed
        df = df.sort_values(by=['item_nbr', 'date'])
        
        for item in df['item_nbr'].unique():
            item_df = df[df['item_nbr'] == item].copy()
            errors = item_df['error'].values
            
            # To get Lead Time uncertainty (7 days), we can sum the errors over 7-day rolling windows
            # because errors might be auto-correlated.
            rolling_7d_error = item_df['error'].rolling(window=7, min_periods=1).sum().dropna().values
            
            stats = {
                'mean_daily_error': np.mean(errors),
                'std_daily_error': np.std(errors),
                'mae_daily_error': np.mean(np.abs(errors)),
                'p10_daily': np.percentile(errors, 10),
                'p50_daily': np.percentile(errors, 50),
                'p90_daily': np.percentile(errors, 90),
                
                # For safety stock at 95% service level, we need the 95th percentile of LEAD TIME (7-day) errors
                'p95_lt_error': np.percentile(rolling_7d_error, 95) if len(rolling_7d_error) > 0 else 0
            }
            self.error_stats[item] = stats
            
    def get_safety_stock(self, item_nbr, lead_time=7, service_level=0.95):
        """
        Returns the empirical safety stock required for a given lead time and service level.
        Calculates the appropriate quantile of the L-day rolling cumulative out-of-sample forecast errors.
        """
        if item_nbr not in self.error_stats:
            return 0
            
        item_df = self.cv_preds[self.cv_preds['item_nbr'] == item_nbr].sort_values('date')
        
        # If lead time is too large for the validation data size, it's unsafe.
        # We need at least one valid rolling window.
        if len(item_df) < lead_time:
            return 0  # Cannot reliably estimate
            
        rolling_error = item_df['error'].rolling(window=lead_time, min_periods=1).sum().dropna().values
        
        quantile_target = service_level * 100
        emp_ss = np.percentile(rolling_error, quantile_target)
        
        return max(0, emp_ss)
        
    def get_daily_bounds(self, item_nbr, forecast):
        """
        Returns (lower_bound, upper_bound) based on P10 and P90 of daily out-of-sample errors.
        lower = forecast + P10_error
        upper = forecast + P90_error
        """
        if item_nbr not in self.error_stats:
            return forecast, forecast
            
        lower = forecast + self.error_stats[item_nbr]['p10_daily']
        upper = forecast + self.error_stats[item_nbr]['p90_daily']
        return max(0, lower), max(0, upper)
        
    def summary_df(self):
        return pd.DataFrame.from_dict(self.error_stats, orient='index')

if __name__ == "__main__":
    unc = EmpiricalUncertainty()
    print(unc.summary_df().head())
