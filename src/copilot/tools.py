import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from uncertainty.empirical import EmpiricalUncertainty
from evaluation.validation import get_test_period

# Load common data once to avoid repeated disk I/O in tools
try:
    df_features = pd.read_csv("data/processed/store_sales_features.csv", parse_dates=['date'])
    cv_preds = pd.read_csv("data/processed/cv_predictions.csv", parse_dates=['date'])
    overall_metrics = pd.read_csv("reports/overall_metrics.csv")
    prod_metrics = pd.read_csv("reports/per_product_metrics.csv")
    replenishment = pd.read_csv("reports/replenishment/replenishment_simulation.csv")
    items = pd.read_csv("data/raw/items.csv")
    
    unc = EmpiricalUncertainty()
    
    # Pre-calculate LightGBM test predictions to serve forecast tools instantly
    from models.forecast_model import ForecastModel
    test_start_str, test_end_str = get_test_period()
    test_start_dt = pd.to_datetime(test_start_str)
    
    train_df = df_features[df_features['date'] < test_start_str].copy()
    test_df = df_features[(df_features['date'] >= test_start_str) & (df_features['date'] <= test_end_str)].copy()
    
    model = ForecastModel()
    model.train(train_df)
    test_df['predicted_demand'] = model.predict(test_df)
    
except Exception as e:
    print(f"Warning: Data not fully initialized for tools. {e}")

def get_product_forecast(item_id: int):
    """
    Returns the next 7-day forecast for a given item, along with empirical uncertainty bounds.
    """
    item_df = test_df[test_df['item_nbr'] == item_id].sort_values('date').head(7)
    if item_df.empty:
        return f"No forecast available for item {item_id}."
        
    forecasts = []
    for _, row in item_df.iterrows():
        pred = row['predicted_demand']
        lower, upper = unc.get_daily_bounds(item_id, pred)
        forecasts.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'forecast': round(pred, 2),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2)
        })
    return {"item_id": item_id, "horizon": "7 days", "forecasts": forecasts}

def get_product_history(item_id: int):
    """
    Returns relevant historical demand information for the 14 days immediately preceding the forecast period.
    """
    hist = train_df[train_df['item_nbr'] == item_id].sort_values('date').tail(14)
    if hist.empty:
        return f"No history found for item {item_id}."
    
    history = []
    for _, row in hist.iterrows():
        history.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'unit_sales': row['unit_sales'],
            'onpromotion': bool(row['onpromotion'])
        })
    return {"item_id": item_id, "recent_history": history}

def get_forecast_accuracy(item_id: int):
    """
    Returns product-level accuracy metrics on the test set, comparing LightGBM vs Baselines.
    """
    metrics = prod_metrics[prod_metrics['item_nbr'] == item_id]
    if metrics.empty:
        return f"No accuracy metrics found for item {item_id}."
    return {"item_id": item_id, "metrics": metrics.to_dict(orient='records')}

def get_promotion_context(item_id: int):
    """
    Returns current/forecast promotion status and historical promotion response.
    """
    # Recent history
    hist = train_df[train_df['item_nbr'] == item_id].tail(365)
    if hist.empty:
        return "No data."
    
    promo_mean = hist[hist['onpromotion'] == True]['unit_sales'].mean()
    nopromo_mean = hist[hist['onpromotion'] == False]['unit_sales'].mean()
    
    # Future promotions
    future = test_df[test_df['item_nbr'] == item_id].head(7)
    future_promos = future['onpromotion'].sum()
    
    return {
        "item_id": item_id,
        "historical_mean_on_promo": round(promo_mean, 2) if pd.notnull(promo_mean) else 0,
        "historical_mean_no_promo": round(nopromo_mean, 2) if pd.notnull(nopromo_mean) else 0,
        "upcoming_7d_promotion_days": int(future_promos)
    }

def get_replenishment_recommendation(item_id: int, current_inventory: float, lead_time: int = 7, service_level: float = 0.95, case_pack_size: int = 1):
    """
    Calculates safety stock, ROP, and ROQ dynamically for a specific product and inventory level.
    """
    import math
    # Dynamic SS using empirical quantiles for L days
    emp_ss = unc.get_safety_stock(item_id, lead_time=lead_time, service_level=service_level)
        
    item_df = test_df[test_df['item_nbr'] == item_id].sort_values('date')
    available_horizon = len(item_df)
    
    if lead_time > available_horizon:
        return {
            "error": "Data Availability Limitation",
            "message": f"Requested lead time of {lead_time} days exceeds the available {available_horizon}-day forecast horizon. Cannot calculate valid expected lead-time demand."
        }
        
    item_df = item_df.head(lead_time)
    ltd_forecast = item_df['predicted_demand'].sum()
    
    rop = ltd_forecast + emp_ss
    raw_roq = max(0, rop - current_inventory)
    
    # Case pack rounding (round UP to nearest multiple of case_pack_size)
    rounded_roq = math.ceil(raw_roq / case_pack_size) * case_pack_size
    
    return {
        "item_id": item_id,
        "assumptions": {"lead_time_days": lead_time, "service_level": service_level, "case_pack_size": case_pack_size},
        "inputs": {"current_inventory": current_inventory},
        "calculation": {
            "expected_lead_time_demand": round(ltd_forecast, 2),
            "safety_stock": round(emp_ss, 2),
            "reorder_point": round(rop, 2),
            "raw_recommended_order_quantity": round(raw_roq, 2),
            "rounded_recommended_order_quantity": round(rounded_roq, 2)
        }
    }

def compare_products(item_ids: list[int]):
    """
    Returns structured forecast/replenishment comparisons across multiple items.
    """
    comparison = []
    for i in item_ids:
        rep = replenishment[replenishment['item_nbr'] == i]
        if not rep.empty:
            r = rep.iloc[0]
            comparison.append({
                "item_id": i,
                "family": r['family'],
                "next_7d_forecast": r['lead_time_demand_forecast'],
                "safety_stock": r['safety_stock'],
                "is_high_risk_lumpy": bool(r['is_high_risk_lumpy'])
            })
    return comparison

def get_highest_uncertainty(top_n: int = 5):
    """
    Returns the products with the highest demand variability, ranked by Coefficient of Variation (CV).
    """
    if 'cv' not in replenishment.columns:
        return "Uncertainty data not available."
    
    highest = replenishment.sort_values(by='cv', ascending=False).head(top_n)
    results = []
    for _, r in highest.iterrows():
        results.append({
            "item_id": int(r['item_nbr']),
            "family": str(r['family']),
            "cv": round(r['cv'], 2),
            "is_high_risk_lumpy": bool(r['is_high_risk_lumpy']),
            "safety_stock": float(r['safety_stock'])
        })
    return {"ranking_metric": "Coefficient of Variation (CV)", "highest_demand_variability_products": results}
