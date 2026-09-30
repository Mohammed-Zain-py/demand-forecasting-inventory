import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

# Configure Streamlit page
st.set_page_config(
    page_title="Demand & Replenishment Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling for Portfolio Quality ---
st.markdown("""
    <style>
    h1, h2, h3 {font-family: 'Inter', sans-serif;}
    .kpi-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
    }
    .kpi-title {font-size: 14px; color: #888888; text-transform: uppercase; font-weight: 600;}
    .kpi-value {font-size: 28px; color: var(--text-color); font-weight: bold; margin-top: 5px;}
    .warning-card {
        background-color: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid rgba(255, 193, 7, 0.3);
        margin-bottom: 20px;
        font-size: 14px;
    }
    .chat-user {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 10px;
    }
    .chat-bot {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin-bottom: 20px; 
        border: 1px solid var(--secondary-background-color);
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    overall_metrics = pd.read_csv("reports/overall_metrics.csv")
    prod_metrics = pd.read_csv("reports/per_product_metrics.csv")
    replenishment = pd.read_csv("reports/replenishment/replenishment_simulation.csv")
    cv_preds = pd.read_csv("data/processed/cv_predictions.csv", parse_dates=['date'])
    df_features = pd.read_csv("data/processed/store_sales_features.csv", parse_dates=['date'])
    items = pd.read_csv("data/raw/items.csv")
    return overall_metrics, prod_metrics, replenishment, cv_preds, df_features, items

try:
    overall_metrics, prod_metrics, replenishment, cv_preds, df_features, items = load_data()
except Exception as e:
    st.error(f"Error loading backend data. Make sure the analytical pipeline has been run. {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("📦 System Navigation")
page = st.sidebar.radio("Select Module:", [
    "1. Executive Overview",
    "2. Product Forecast",
    "3. Replenishment Planning",
    "4. Model Performance",
    "5. Ask the App"
])

st.sidebar.markdown("---")

# --- Helper Functions ---
@st.cache_resource
def get_trained_model():
    from src.evaluation.validation import get_test_period
    from src.models.forecast_model import ForecastModel
    
    test_start, _ = get_test_period()
    test_start_dt = pd.to_datetime(test_start)
    train_full = df_features[df_features['date'] < test_start_dt].copy()
    
    model = ForecastModel()
    model.train(train_full)
    return model

def plot_forecast(item_id):
    from src.evaluation.validation import get_test_period
    from src.uncertainty.empirical import EmpiricalUncertainty
    
    unc = EmpiricalUncertainty()
    test_start, test_end = get_test_period()
    test_start_dt = pd.to_datetime(test_start)
    
    train = df_features[(df_features['item_nbr'] == item_id) & (df_features['date'] < test_start_dt)].tail(30)
    test = df_features[(df_features['item_nbr'] == item_id) & (df_features['date'] >= test_start_dt)].head(7).copy()
    
    # Needs forecast
    model = get_trained_model()
    test['predicted'] = model.predict(test)
    
    test['lower'] = test['predicted'].apply(lambda x: unc.get_daily_bounds(item_id, x)[0])
    test['upper'] = test['predicted'].apply(lambda x: unc.get_daily_bounds(item_id, x)[1])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train['date'], train['unit_sales'], label='Historical Actuals', color='#2c3e50', marker='o')
    ax.plot(test['date'], test['predicted'], label='Forecast', color='#e74c3c', marker='s', linestyle='--')
    ax.fill_between(test['date'], test['lower'], test['upper'], color='#e74c3c', alpha=0.2, label='80% Empirical Interval')
    
    ax.set_title(f"30-Day History & 7-Day Forecast (One-Step-Ahead) (Item {item_id})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, test

# --- Pages ---
if page == "1. Executive Overview":
    st.title("Demand Forecasting & Replenishment Planning")
    
    st.markdown("### System Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Store Evaluated</div><div class="kpi-value">Store 25</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Products</div><div class="kpi-value">{len(replenishment)}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Forecast Model</div><div class="kpi-value">LightGBM</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Simulated Lead Time</div><div class="kpi-value">7 Days</div></div>', unsafe_allow_html=True)

    st.markdown("### Global Performance (WMAPE)")
    test_metrics = overall_metrics[overall_metrics['Fold'] == 'Test']
    lgbm_wmape = test_metrics[test_metrics['Model'] == 'LightGBM']['WMAPE'].values[0]
    lstm_wmape = test_metrics[test_metrics['Model'] == 'PyTorch LSTM']['WMAPE'].values[0]
    sn_wmape = test_metrics[test_metrics['Model'] == 'Seasonal Naive']['WMAPE'].values[0]
    naive_wmape = test_metrics[test_metrics['Model'] == 'Naive']['WMAPE'].values[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">LightGBM (Prod) WMAPE</div><div class="kpi-value">{lgbm_wmape:.2%}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">PyTorch LSTM WMAPE</div><div class="kpi-value">{lstm_wmape:.2%}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Seasonal Naive WMAPE</div><div class="kpi-value">{sn_wmape:.2%}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Naive WMAPE</div><div class="kpi-value">{naive_wmape:.2%}</div></div>', unsafe_allow_html=True)

    st.markdown("### 23-Product Replenishment Portfolio")
    # Show high risk
    def highlight_risk(s):
        return ['background-color: rgba(255, 75, 75, 0.2)' if v == 'Yes' else '' for v in s]
    
    disp_df = replenishment[['item_nbr', 'family', 'lead_time_demand_forecast', 'safety_stock', 'reorder_point', 'is_high_risk_lumpy']].copy()
    disp_df.rename(columns={'is_high_risk_lumpy': 'High Risk'}, inplace=True)
    disp_df['High Risk'] = disp_df['High Risk'].map({True: 'Yes', False: 'No'})
    
    st.dataframe(disp_df.style.apply(highlight_risk, subset=['High Risk']))

elif page == "2. Product Forecast":
    st.title("Product Forecast & Uncertainty")
    
    item_id = st.selectbox("Select Product", replenishment['item_nbr'].unique())
    
    fam = replenishment[replenishment['item_nbr'] == item_id]['family'].values[0]
    st.subheader(f"Item {item_id} ({fam})")
    
    with st.spinner("Generating prediction intervals..."):
        fig, test_df = plot_forecast(item_id)
        st.pyplot(fig)
        
    st.markdown("#### Numerical Forecast (One-Step-Ahead for Next 7 Days)")
    st.caption("Forecast Interval / Uncertainty Interval: A range showing where actual demand may fall based on past forecast errors. (Note: These are one-step-ahead predictions evaluated over the 7-day test period.)")
    show_df = test_df[['date', 'predicted', 'lower', 'upper', 'onpromotion']].copy()
    show_df['date'] = show_df['date'].dt.strftime('%Y-%m-%d')
    show_df.columns = ['Date', 'Expected Demand', 'Lower Bound', 'Upper Bound', 'On Promotion']
    st.dataframe(show_df)

elif page == "3. Replenishment Planning":
    st.title("Replenishment Decision Simulator")
    
    col1, spacer, col2 = st.columns([1, 0.15, 2])
    
    with col1:
        st.markdown("### Simulation Inputs")
        item_id = st.selectbox("Select Product", replenishment['item_nbr'].unique())
        st.info("Input your current on-hand inventory to simulate a reorder recommendation.")
        st.caption("Note: Inventory values are simulated because the source dataset does not contain historical on-hand inventory or stockout records.")
        current_inv = st.number_input("Simulated Current Inventory", min_value=0, value=20, step=1)
        lead_time = st.number_input("Lead Time (Days)", min_value=1, value=7, step=1)
        service_level_pct = st.number_input("Service Level Target (%)", min_value=1, max_value=99, value=95, step=1)
        case_pack = st.number_input("Case Pack Size", min_value=1, value=1, step=1)
        service_level = service_level_pct / 100.0
        
        st.markdown("---")
        st.markdown("#### 📖 Data Dictionary")
        st.markdown("- **Observed**: Historical actual sales (from Favorita)")
        st.markdown("- **Predicted**: Model forecast (from LightGBM)")
        st.markdown("- **Simulated**: Current inventory")
        st.markdown("- **Assumed**: Lead time, service level, case pack size")
        
    with col2:
        st.markdown("### Replenishment Math")
        from src.copilot.tools import get_replenishment_recommendation
        
        res = get_replenishment_recommendation(item_id, current_inv, lead_time, service_level, case_pack)['calculation']
        
        st.markdown(f"""
        **1. Expected Lead-Time Demand (LTD)**: `{res['expected_lead_time_demand']}`  
        *Expected demand while waiting for new stock to arrive.*
        
        **2. Safety Stock (SS)**: `{res['safety_stock']}`  
        *Extra stock kept to protect against unexpected demand.*  
        *(Technical: based on historical forecast errors during the selected lead time.)*
        
        **3. Reorder Point (ROP)**: `{res['reorder_point']}`  
        *The inventory level at which we should reorder.*  
        *(Technical: Expected LTD + Safety Stock)*
        
        **4. Current Inventory**: `{current_inv}`  
        
        **5. Raw Recommended Order Quantity (ROQ)**: `{res['raw_recommended_order_quantity']}`  
        *How much stock the system recommends ordering.*  
        *(Technical: max(0, ROP - Current Inventory))*
        
        **6. Rounded Recommended Order Quantity**: `{res['rounded_recommended_order_quantity']}`  
        *(Rounded UP to nearest case pack size of {case_pack})*
        """)
        
        st.markdown("---")
        st.metric(label="Action: Order Quantity", value=f"{res['rounded_recommended_order_quantity']} units")
        if replenishment[replenishment['item_nbr'] == item_id]['is_high_risk_lumpy'].values[0]:
            st.error("⚠️ HIGH RISK: This product has a high Coefficient of Variation (>1.5). Standard min-max policies may over-order. Review manually.")

elif page == "4. Model Performance":
    st.title("Model Validation & Errors")
    
    st.markdown("### Chronological Folds Performance (WMAPE)")
    folds = overall_metrics[overall_metrics['Fold'] != 'Test'].copy()
    st.bar_chart(folds, x='Fold', y='WMAPE', color='Model', stack=False)
    
    st.markdown("### Final Test Set Performance")
    test_set = overall_metrics[overall_metrics['Fold'] == 'Test'].copy()
    st.dataframe(test_set[['Model', 'MAE', 'RMSE', 'WMAPE']].sort_values('WMAPE').reset_index(drop=True))
    
    st.markdown("### The Item 257847 Challenge")
    st.warning("No model is perfect. Item 257847 shows a significant performance gap between LightGBM and Seasonal Naive in the final test period.")
    st.markdown("""
    **Diagnosis:**
    - Item 257847 is NOT highly intermittent (zeros are ~10%).
    - However, it is highly erratic. Mean demand is ~97 units, but during August 2017 (the final test set), Sales dropped sharply to zero on 14 of the 15 days in the final test period.
    - The ML model relied on moving averages and lagged structural data, so it persistently predicted ~10-20 units daily, failing to adapt to a sudden 100% drop in volume.
    - **Lesson**: Tree-based models optimize global MSE and struggle with sudden structural breaks in individual product histories.
    """)

elif page == "5. Ask the App":
    st.title("💬 Ask the App")
    st.markdown("Ask questions about forecasts, inventory, uncertainty, and replenishment.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi. Ask me about the forecasts, historical demand, uncertainty estimates, or replenishment calculations.\n\nExamples:\n- *Why is the recommended order quantity for item 105575 30 units?*\n- *Which products have the highest demand variability?*\n- *What is the forecast for item 129635?*"}
        ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><b>App:</b> {message["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question about your inventory..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-user"><b>You:</b> {prompt}</div>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing data..."):
            from src.copilot.agent import ask_copilot
            response = ask_copilot(prompt)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f'<div class="chat-bot"><b>App:</b> {response}</div>', unsafe_allow_html=True)

