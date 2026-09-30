import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.copilot.tools import get_replenishment_recommendation, get_product_forecast, get_highest_uncertainty

def run_tests():
    print("Testing get_replenishment_recommendation for Item 129635...")
    res = get_replenishment_recommendation(129635, current_inventory=43)
    assert res['calculation']['reorder_point'] > 0
    assert res['calculation']['raw_recommended_order_quantity'] == max(0, res['calculation']['reorder_point'] - 43)
    print("Pass: ROP and ROQ logic matches backend.")

    print("Testing get_product_forecast for Item 129635...")
    forecast = get_product_forecast(129635)
    assert len(forecast['forecasts']) == 7
    assert forecast['forecasts'][0]['lower_bound'] < forecast['forecasts'][0]['upper_bound']
    print("Pass: Forecast returns 7 days with valid bounds.")
    
    print("Testing robustness on fake item 999999...")
    res_fake = get_replenishment_recommendation(999999, 10)
    assert 'error' in res_fake
    print("Pass: Handled unknown item.")

    print("Testing get_replenishment_recommendation horizon limits...")
    res_15 = get_replenishment_recommendation(129635, 10, lead_time=15)
    assert 'error' not in res_15
    assert res_15['assumptions']['lead_time_days'] == 15
    
    res_16 = get_replenishment_recommendation(129635, 10, lead_time=16)
    assert 'error' in res_16
    assert res_16['error'] == "Data Availability Limitation"
    print("Pass: Horizon limits enforced.")

    print("Testing get_highest_uncertainty...")
    unc_ranking = get_highest_uncertainty(3)
    prods = unc_ranking['highest_demand_variability_products']
    assert len(prods) == 3
    # Verify sorting
    assert prods[0]['cv'] >= prods[1]['cv'] >= prods[2]['cv']
    # Verify exact structure
    assert 'item_id' in prods[0] and 'cv' in prods[0]
    print("Pass: Demand variability ranking tool works and is sorted correctly.")

if __name__ == "__main__":
    run_tests()
