# Demand Forecasting and Replenishment Planning System with a Natural Language Interface

## 1. Overview
Retail stores need to anticipate product demand and determine how much stock should be planned for upcoming demand.

This system forecasts product-level demand for a real retail store, estimates forecast uncertainty, converts forecasts into a configurable replenishment simulation, and provides a domain-specific LLM-based interface that explains the resulting forecasts and replenishment decisions.

*Note: This is a simulation based on historical sales. It does not perform real-world inventory optimization because true on-hand inventory levels are not present in the dataset.*

## 2. Key Features
- **Real retail sales data**: Based on the Kaggle Corporación Favorita Grocery Sales dataset.
- **Product-level demand forecasting**: Forecasts daily demand for individual items.
- **Leakage-safe lag/rolling features**: Prevents future information from bleeding into historical training rows.
- **Baselines & ML Models**: Evaluates Naive, Seasonal Naive, LightGBM, and PyTorch LSTM global forecasting models.
- **Rolling chronological validation**: Tests models on 6 consecutive chronological folds, each 30 days long, strictly ordered before the final test period.
- **Empirical prediction intervals**: Computes out-of-sample error distributions to bound uncertainty.
- **Replenishment Simulation**: Dynamically calculates Reorder Point (ROP) and Recommended Order Quantity (ROQ) based on Expected Lead-Time Demand and Safety Stock.
- **Configurable assumptions**: Allows dynamic adjustment of simulated inventory, lead time, service level target, and case-pack rounding.
- **Streamlit dashboard**: A portfolio-quality visual interface.
- **Tool-grounded Gemini Interface**: A generative AI agent strictly constrained to deterministic backend math.

## 3. Architecture

```text
Data (Favorita Parquet)
   │
   └──► Preprocessing (Complete missing dates / identify store closures)
          │
          └──► Feature Engineering (Lags, rolling stats, calendar, promos)
                 │
                 └──► Forecasting (LightGBM & PyTorch LSTM) & Validation (Chronological folds)
                        │
                        └──► Uncertainty (Empirical Out-Of-Sample Quantiles)
                               │
                               └──► Replenishment (LTD + Safety Stock = ROP)
                                      │
                                      └──► Dashboard (Streamlit UI)
                                             └──► Ask the App (Tool-Grounded LLM)
```

The Ask the App interface sits **above** the deterministic analytical backend. It does not calculate numbers; it invokes backend Python tools to retrieve ML predictions and inventory logic, then explains the reasoning in natural language.

## 4. Dataset
The project uses a subset of the **Corporación Favorita Grocery Sales dataset**:
- **Store**: 25
- **Products**: 23 selected items across various families (Beverages, Grocery, Produce, etc.)
- **Date Range**: 2013-01-01 to 2017-08-15
- **Features**: `date`, `store_nbr`, `item_nbr`, `unit_sales`, `onpromotion`

**Why only one store and 23 products?**
This is an MVP scope designed to demonstrate the complete ML-to-Supply-Chain pipeline without processing the entire 900MB competition dataset. Store 25 is not claimed to be representative of all stores.

## 5. Data Challenges
- **Implicit zero-sales rows**: The raw dataset only records rows when a sale occurred. Missing dates were re-indexed and filled with `0` sales.
- **Store closures**: Dates with zero sales across all products were verified as store closures and flagged to prevent polluting the demand history.
- **Missing promotion tracking**: Early years lacked promotion data. These were imputed carefully with a `promotion_known` boolean flag.
- **Lack of historical inventory & lost sales**: We do not know when the store actually stocked out. Therefore, stockouts cannot be historically modeled.

## 6. Forecasting Methodology

### Baselines
- **Naive**: Predicts tomorrow will equal today.
- **Seasonal Naive**: Predicts tomorrow will equal the same day last week.

### ML Models
- **LightGBM**: A gradient boosting tree model trained globally across all 23 products simultaneously. It is the primary production forecasting model used by the replenishment backend.
- **PyTorch LSTM**: A sequence-based deep learning model using a 28-day historical lookback window. It is evaluated as a comparison model in the forecasting pipeline.

### Features
- **Lags**: 1-day, 7-day, 14-day, and 28-day lag features.
- **Rolling statistics**: 7-day, 14-day, and 28-day rolling means, plus 7-day and 28-day rolling standard deviations. These are calculated from shifted historical sales so the prediction date itself is not included.
- **Calendar features**: Day of week, day of month, week of year, month, quarter, and weekend indicator.
- **Exogenous**: `onpromotion`, `promotion_known`, `store_closed`.
- **Product identity**: `item_nbr` is used by LightGBM as a categorical feature. The LSTM uses the temporal feature sequence rather than a direct `item_nbr` input.

### Leakage Prevention
Lag and rolling features only use information available strictly before the prediction date. The first 28 days of each product's history were excluded because the longest required historical feature window is 28 days.

## 7. Model Evaluation
Final Test Set (August 1, 2017 to August 15, 2017):

| Model | MAE | RMSE | WMAPE |
| :--- | :--- | :--- | :--- |
| Naive | 8.70 | 22.84 | 64.26% |
| Seasonal Naive | 7.06 | 18.19 | 52.14% |
| **PyTorch LSTM** | **5.31** | **10.26** | **39.19%** |
| **LightGBM** | **5.91** | **11.74** | **43.60%** |

Validation Mean (across 6 historical folds, 30 days each):
- Naive WMAPE: 65.61%
- Seasonal Naive WMAPE: 68.29%
- **PyTorch LSTM WMAPE**: **47.13%**
- **LightGBM WMAPE**: **45.15%**

Model Selection Conclusion: While the PyTorch LSTM outperformed LightGBM in the final 15-day test set (39.19% vs 43.60%), LightGBM demonstrated lower mean WMAPE across the expanded 6-fold chronological validation (45.15% vs 47.13%). Given this broader validation result and its role in the existing replenishment backend, LightGBM is retained as the primary production forecasting engine, while PyTorch LSTM remains in the pipeline as a deep-learning comparison model.

Note on Dashboard Forecasts: The interactive 7-day forecast presented in the dashboard utilizes one-step-ahead (teacher-forced) predictions over the evaluation period rather than a pure recursive multi-step projection.

## 8. Forecast Uncertainty
Uncertainty is calculated using empirical out-of-sample forecast errors from the cross-validation predictions rather than assuming a parametric normal error distribution.
- **Nominal target**: 80% empirical interval.
- **Observed coverage**: Empirical interval with approximately 78.36% observed coverage on the evaluated validation predictions.

## 9. Replenishment Simulation
The replenishment recommendation transforms the forecast into a business action.

**Formulas**:
- **Expected LTD** (Lead-Time Demand) = $\sum_{t=1}^{L} \text{Forecast}_t$
- **Safety Stock** = $P_{\text{service\_level}}(\text{Cumulative L-day Forecast Errors})$
- **ROP** (Reorder Point) = $\text{LTD} + \text{Safety Stock}$
- **Raw ROQ** (Recommended Order Quantity) = $\max(0, \text{ROP} - \text{Current Inventory})$
- **Rounded ROQ** = $\lceil \text{Raw ROQ} / \text{Case Pack} \rceil \times \text{Case Pack}$

**Variables**:
- *Lead time* = Simulation assumption (dynamic)
- *Service level* = Simulation target (dynamic)
- *Current inventory* = User-provided simulated input
- *Case pack size* = User-configurable simulation assumption

**Data availability constraint:** The current evaluation/test horizon contains 15 days (August 1–15, 2017). The replenishment interface therefore rejects lead times greater than 15 days rather than calculating an LTD from an incomplete forecast horizon.

## 10. Natural Language Interface
The LLM interface is NOT a generic chatbot. It is a strictly grounded LLM architecture.

**Tools exposed to the LLM:**
- `get_product_forecast(item_id)`
- `get_product_history(item_id)`
- `get_forecast_accuracy(item_id)`
- `get_promotion_context(item_id)`
- `get_replenishment_recommendation(item_id, current_inventory, lead_time, service_level, case_pack_size)`
- `compare_products(item_ids)`
- `get_highest_uncertainty(top_n)`

The `get_highest_uncertainty` tool ranks products by **Coefficient of Variation (CV)**, which represents demand variability. It should not be interpreted as a direct forecast-error uncertainty ranking.

**Workflow**:
LLM → invokes Python tools → retrieves deterministic backend JSON → explains reasoning in natural language. Unavailable information (actual inventory, stockouts) is explicitly flagged and safely declined.

## 11. Example LLM Questions
- *"Why is Item 129635 recommended for replenishment?"*
- *"How accurate is the forecast for Item 273528?"*
- *"Which products have the highest demand variability?"*
- *"What happens if lead time increases from 7 to 10 days?"*
- *"Compare the replenishment requirements of the beverage products."*

## 12. Limitations
- **No real on-hand inventory**: This is a simulated exercise.
- **No historical stockout/lost-sales data**: True lost demand cannot be calculated.
- **Replenishment is a simulation**: Safety stock triggers and Reorder quantities assume the simulated state is reality.
- **Assumptions**: Lead times and case packs are assumptions, not derived from real supplier data.
- **Forecast intervals are empirical, not guaranteed**: They represent past out-of-sample variance, not absolute statistical guarantees.
- **Service-level backtest**: The observed service-level performance is not expected to exactly match the nominal target because the backtest is based on a limited historical evaluation window and empirical error quantiles.
- **Scope**: Single-store MVP limited to 23 products.
- **Model weakness**: Abrupt structural changes in demand can still challenge the forecasting models.

## 13. Project Structure
```
project/
├── data/
│   ├── raw/                 # Store 25 subset (items.csv, stores.csv, store_sales.csv)
│   └── processed/           # Processed features and CV predictions
├── src/
│   ├── copilot/             # LLM agent and deterministic tools
│   ├── dashboard/           # Streamlit application
│   ├── data/                # Preprocessing pipeline
│   ├── evaluation/          # Validation and metric generation
│   ├── features/            # Feature engineering
│   ├── models/              # LightGBM, LSTM, and baselines
│   │   ├── baseline.py
│   │   ├── forecast_model.py
│   │   └── lstm_forecast.py
│   ├── replenishment/       # Simulation logic
│   ├── uncertainty/         # Empirical uncertainty logic
│   └── visualization/       # EDA plotting scripts
├── reports/                 # Output CSVs and PNGs
├── tests/                   # Python test scripts
├── .env                     # Environment variable 
├── .gitignore
├── README.md
└── requirements.txt         # Python dependencies
```

## 14. Installation
```bash
git clone https://github.com/Mohammed-Zain-py/demand-forecasting-inventory.git
cd demand-forecasting-inventory
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## 15. Environment Variables
To enable the Ask the App tab in the dashboard, create a `.env` file and provide your API key.
```bash
GEMINI_API_KEY=your_api_key_here
```
*Never commit your `.env` file to version control.*

## 16. Testing

Run the project test suites with:

```bash
python tests/test_tools.py
python tests/test_pipeline.py
python -m unittest discover tests/
```

The tests cover replenishment calculations, dynamic lead-time and service-level behavior, case-pack rounding, data-availability boundaries, uncertainty ranking, unknown-product handling, and adversarial LLM grounding checks.
