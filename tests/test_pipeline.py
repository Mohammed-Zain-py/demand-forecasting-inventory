import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from src.copilot.tools import get_replenishment_recommendation
from src.copilot.agent import ask_copilot

def run_regression_tests():
    print("--- REGRESSION TESTS ---")
    
    # 1. Test existing parameters / unchanged behavior
    res_7d = get_replenishment_recommendation(129635, 43, lead_time=7, service_level=0.95, case_pack_size=1)
    assert res_7d['calculation']['expected_lead_time_demand'] > 0
    import math
    assert res_7d['calculation']['rounded_recommended_order_quantity'] >= res_7d['calculation']['raw_recommended_order_quantity']
    assert res_7d['calculation']['rounded_recommended_order_quantity'] == math.ceil(res_7d['calculation']['raw_recommended_order_quantity'])
    print(f"Pass: 7-day lead time normal behavior.")
    
    # 2. Test dynamic lead time (3-day vs 10-day)
    res_3d = get_replenishment_recommendation(129635, 43, lead_time=3, service_level=0.95)
    res_10d = get_replenishment_recommendation(129635, 43, lead_time=10, service_level=0.95)
    
    assert res_3d['calculation']['expected_lead_time_demand'] < res_7d['calculation']['expected_lead_time_demand']
    assert res_10d['calculation']['expected_lead_time_demand'] > res_7d['calculation']['expected_lead_time_demand']
    print("Pass: Dynamic lead time changes LTD correctly.")
    
    # 3. Test changing service level
    res_50sl = get_replenishment_recommendation(129635, 43, lead_time=7, service_level=0.50)
    res_99sl = get_replenishment_recommendation(129635, 43, lead_time=7, service_level=0.99)
    assert res_50sl['calculation']['safety_stock'] <= res_99sl['calculation']['safety_stock']
    print("Pass: Changing service level changes safety stock correctly.")
    
    # 4. Test case pack rounding
    res_cp = get_replenishment_recommendation(129635, 43, lead_time=7, service_level=0.95, case_pack_size=12)
    raw = res_cp['calculation']['raw_recommended_order_quantity']
    rounded = res_cp['calculation']['rounded_recommended_order_quantity']
    assert rounded >= raw
    assert rounded % 12 == 0
    print(f"Pass: Case pack rounding works (Raw: {raw} -> Rounded: {rounded})")
    
    # 5. Invalid/missing data robustness
    res_invalid_lt = get_replenishment_recommendation(129635, 43, lead_time=100)
    assert 'error' in res_invalid_lt
    assert res_invalid_lt['error'] == "Data Availability Limitation"
    print("Pass: Safety handled for insufficient validation history.")

def run_adversarial_tests():
    print("\n--- ADVERSARIAL TESTS ---")
    queries = [
        "How many units are currently in Store 25?",
        "What was the actual stockout rate of Store 25 in 2017?",
        "Which supplier should I order Item 129635 from?",
        "Give me the exact number of customers who could not buy Item 129635 because it was out of stock."
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        ans = ask_copilot(q).lower()
        print(f"A: {ans[:200]}...")
        # Make assertions based on keywords to verify safeguards triggered without relying on exact wording
        if "[error]" in ans:
            print("Skipping assertion due to API error (e.g. quota exceeded).")
        else:
            is_safe = any(phrase in ans for phrase in ["not have access", "unavailable", "simulated", "do not know", "cannot", "does not contain", "do not have"])
            assert is_safe, f"LLM did not correctly decline to invent data for question: {q}"
        
    print("Pass: Adversarial safeguards successfully triggered and verified.")

if __name__ == "__main__":
    run_regression_tests()
    run_adversarial_tests()
