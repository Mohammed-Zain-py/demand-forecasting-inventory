import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from uncertainty.empirical import EmpiricalUncertainty
from evaluation.validation import get_test_period

class ReplenishmentSimulation:
    def __init__(self, lead_time=7, service_level=0.95):
        self.lead_time = lead_time
        self.service_level = service_level
        self.uncertainty = EmpiricalUncertainty()
        self.df_features = pd.read_csv("data/processed/store_sales_features.csv", parse_dates=['date'])
        
    def run_simulation(self):
        from models.forecast_model import ForecastModel
        test_start_str, test_end_str = get_test_period()
        test_start_dt = pd.to_datetime(test_start_str)
        
        train_df = self.df_features[self.df_features['date'] < test_start_str].copy()
        test_df = self.df_features[(self.df_features['date'] >= test_start_str) & (self.df_features['date'] <= test_end_str)].copy()
        
        model = ForecastModel()
        model.train(train_df)
        test_df['predicted_demand'] = model.predict(test_df)
        
        lt_df = test_df[test_df['date'] < test_start_dt + pd.Timedelta(days=self.lead_time)].copy()
        
        items_meta = pd.read_csv("data/raw/items.csv")
        results = []
        
        for item in lt_df['item_nbr'].unique():
            item_df = lt_df[lt_df['item_nbr'] == item]
            family = items_meta[items_meta['item_nbr'] == item]['family'].values[0]
            
            ltd_forecast = item_df['predicted_demand'].sum()
            
            next_day_forecast = item_df.iloc[0]['predicted_demand']
            lower_bound, upper_bound = self.uncertainty.get_daily_bounds(item, next_day_forecast)
            
            safety_stock = self.uncertainty.get_safety_stock(item)
            reorder_point = ltd_forecast + safety_stock
            
            np.random.seed(item)
            simulated_inv = max(0, int(ltd_forecast * 0.5) + np.random.randint(0, 10))
            
            roq = max(0, reorder_point - simulated_inv)
            
            # Identify lumpy/intermittent demand using Coefficient of Variation (CV)
            train_item = train_df[train_df['item_nbr'] == item]
            mean_d = train_item['unit_sales'].mean()
            std_d = train_item['unit_sales'].std()
            cv = std_d / mean_d if mean_d > 0 else 0
            
            # Flag high CV (> 1.5) or manually flag specific user-identified high-risk items (e.g. 257847)
            is_intermittent = (cv > 1.5) or (item == 257847)
            
            results.append({
                'item_nbr': item,
                'family': family,
                'next_day_forecast': round(next_day_forecast, 2),
                'next_day_lower': round(lower_bound, 2),
                'next_day_upper': round(upper_bound, 2),
                'lead_time_demand_forecast': round(ltd_forecast, 2),
                'safety_stock': round(safety_stock, 2),
                'reorder_point': round(reorder_point, 2),
                'simulated_current_inv': simulated_inv,
                'recommended_order_quantity': round(roq, 2),
                'is_high_risk_lumpy': is_intermittent,
                'cv': round(cv, 2)
            })
            
        results_df = pd.DataFrame(results)
        results_df.to_csv("reports/replenishment/replenishment_simulation.csv", index=False)
        
        print(f"Replenishment simulation saved to reports/replenishment/replenishment_simulation.csv")
        print("\n--- Flagged High-Risk/Lumpy Items ---")
        print(results_df[results_df['is_high_risk_lumpy'] == True][['item_nbr', 'family', 'cv', 'lead_time_demand_forecast', 'safety_stock', 'recommended_order_quantity']].to_string(index=False))
        
        # Validation checks
        print("\n--- Validation of Uncertainty Intervals ---")
        cv_preds = self.uncertainty.cv_preds
        cv_preds['daily_lower'] = cv_preds.apply(lambda r: self.uncertainty.get_daily_bounds(r['item_nbr'], r['predicted'])[0], axis=1)
        cv_preds['daily_upper'] = cv_preds.apply(lambda r: self.uncertainty.get_daily_bounds(r['item_nbr'], r['predicted'])[1], axis=1)
        
        cv_preds['in_bound'] = (cv_preds['unit_sales'] >= cv_preds['daily_lower']) & (cv_preds['unit_sales'] <= cv_preds['daily_upper'])
        print(f"Empirical Nominal Interval: 80% (P10 to P90)")
        print(f"Observed Coverage: {cv_preds['in_bound'].mean():.2%}")
        avg_width = (cv_preds['daily_upper'] - cv_preds['daily_lower']).mean()
        print(f"Average Interval Width: {avg_width:.2f} units")

if __name__ == "__main__":
    sim = ReplenishmentSimulation(lead_time=7, service_level=0.95)
    sim.run_simulation()
