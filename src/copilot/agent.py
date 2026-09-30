import os
import json
import traceback
import sys

from dotenv import load_dotenv
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env')))

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from google import genai
from google.genai import types

from src.copilot.tools import (
    get_product_forecast,
    get_product_history,
    get_forecast_accuracy,
    get_promotion_context,
    get_replenishment_recommendation,
    compare_products,
    get_highest_uncertainty
)

API_KEY = os.environ.get("GEMINI_API_KEY")
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY)

tools = [
    get_product_forecast,
    get_product_history,
    get_forecast_accuracy,
    get_promotion_context,
    get_replenishment_recommendation,
    compare_products,
    get_highest_uncertainty
]

SYSTEM_PROMPT = """
You are a domain-specific Forecasting and Replenishment Assistant.
You have direct access to backend analytical tools that provide LightGBM ML forecasts, empirical safety stocks, and replenishment recommendations.

Rules:
1. ALWAYS use the provided tools to answer data questions. DO NOT invent or calculate numbers yourself if a tool can provide them.
2. The user will often ask "Why is item X recommended for Y units?". You must run `get_replenishment_recommendation` (and optionally `get_product_forecast`) to explain the exact math: Expected LTD + Safety Stock = ROP. ROQ = max(0, ROP - Current Inventory).
3. Distinguish clearly between OBSERVED (historical actuals), PREDICTED (forecast), SIMULATED (current inventory), and ASSUMED (lead time / service level).
4. Do NOT claim access to real supplier systems, real stockouts, or live inventory. Emphasize that inventory values are SIMULATED.
5. Do NOT just say "Demand is high". Explain using the actual numbers returned by the tools.
6. If a tool returns an error or "No data", politely tell the user the data is unavailable.
"""

def ask_copilot(user_query: str) -> str:
    if not API_KEY or client is None:
        return (
            "**Gemini API Key Missing.**\n\n"
            "I am ready to help, but I need a `GEMINI_API_KEY` environment variable to operate. "
            "However, my backend tools are fully functional. For example, if you asked about item 129635, "
            f"the backend calculates:\n\n```json\n{json.dumps(get_replenishment_recommendation(129635, 43), indent=2)}\n```"
        )
        
    try:
        chat = client.chats.create(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=SYSTEM_PROMPT
            )
        )
        response = chat.send_message(user_query)
        return response.text
        
    except Exception as e:
        return f"[Error] LLM Error: {str(e)}\n\n{traceback.format_exc()}"

if __name__ == "__main__":
    print("Testing tools directly without LLM:")
    print(get_replenishment_recommendation(129635, 43))
