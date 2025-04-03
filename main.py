# main.py
import os
import json
import re
import sqlite3
from typing import Type

# --- CrewAI, LangChain Imports ---
from crewai import Agent as CrewAgent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_ollama import OllamaLLM

# Import our custom agent classes
from agents import DemandForecastAgent, InventoryAgent, PricingAgent
# Import database utilities
import db_utils

# --- Configure the Ollama LLM ---
print("Configuring Ollama LLM...")
try:
    ollama_llm = OllamaLLM(
        model="ollama/gemma:2b",
        base_url="http://localhost:11434"
    )
    print(f"Ollama LLM configured with model '{ollama_llm.model}' at base URL '{ollama_llm.base_url}'.")
except Exception as e:
    print(f"FATAL ERROR: Failed to configure Ollama LLM. Is Ollama running? Error: {e}")
    exit()

# --- Instantiate Our Custom Agent Logic Classes ---
try:
    print("Initializing custom agents...")
    demand_agent_logic = DemandForecastAgent()
    inventory_agent_logic = InventoryAgent()
    pricing_agent_logic = PricingAgent(base_margin_percent=15.0)
    print("Custom agents initialized successfully.")
except ConnectionError as e:
    print(f"FATAL ERROR during agent initialization: {e}")
    exit()
except Exception as e:
    print(f"FATAL UNEXPECTED ERROR during agent initialization: {e}")
    exit()

# --- Define Custom Tool Classes ---
print("Defining custom tool classes...")

class DemandForecastTool(BaseTool):
    name: str = "DemandForecastTool"
    description: str = "Generates a demand forecast."

    def _run(self, product_id: str, store_id: str, forecast_horizon_days: int) -> str:
        print(f"DEBUG: DemandForecastTool invoked with: product_id={product_id}, store_id={store_id}, forecast_horizon_days={forecast_horizon_days}")
        try:
            prediction, raw_response = demand_agent_logic.generate_forecast(product_id, store_id, forecast_horizon_days)
            print(f"DEBUG: DemandForecastTool result: prediction={prediction}, raw_response={raw_response}")
            return str(prediction) if prediction is not None else f"Error: Failed to generate forecast. Raw response: {raw_response}"
        except Exception as e:
            print(f"ERROR in DemandForecastTool: {e}")
            return f"Error executing demand forecast tool: {e}"

class InventoryCheckTool(BaseTool):
    name: str = "InventoryCheckTool"
    description: str = "Checks inventory status."

    def _run(self, product_id: str, store_id: str) -> str:
        print(f"DEBUG: InventoryCheckTool invoked with: product_id={product_id}, store_id={store_id}")
        try:
            status = inventory_agent_logic.check_stock_status(product_id, store_id)
            print(f"DEBUG: InventoryCheckTool result: {status}")
            return json.dumps(status)
        except Exception as e:
            print(f"ERROR in InventoryCheckTool: {e}")
            return f'{{"error": "Failed to execute inventory check tool: {e}"}}'

class PricingRecommendationTool(BaseTool):
    name: str = "PricingRecommendationTool"
    description: str = "Recommends pricing."

    def _run(self, product_id: str, store_id: str) -> str:
        print(f"DEBUG: PricingRecommendationTool invoked with: product_id={product_id}, store_id={store_id}")
        try:
            recommendation = pricing_agent_logic.recommend_price(product_id, store_id)
            print(f"DEBUG: PricingRecommendationTool result: {recommendation}")
            result = {
                "price": recommendation.get("final_price"),
                "discount_percentage": recommendation.get("final_discount")
            }
            return json.dumps(result)
        except Exception as e:
            print(f"ERROR in PricingRecommendationTool: {e}")
            return f'{{"error": "Failed to execute pricing recommendation tool: {e}"}}'

print("Custom tool classes defined.")

# --- Define CrewAI Agents ---
print("Defining CrewAI agents...")
demand_forecaster = CrewAgent(
    role="Demand Forecasting Analyst",
    goal="Generate demand forecasts.",
    backstory="Analytical AI for demand prediction.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
    tools=[DemandForecastTool()]
)

inventory_manager = CrewAgent(
    role="Inventory Management Specialist",
    goal="Check inventory status.",
    backstory="AI for stock monitoring.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
    tools=[InventoryCheckTool()]
)

pricing_strategist = CrewAgent(
    role="Retail Pricing Strategist",
    goal="Recommend pricing.",
    backstory="AI for pricing strategies.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm,
    tools=[PricingRecommendationTool()]
)
print("CrewAI agents defined.")

# --- Define Tasks ---
print("Defining tasks...")
product_id_input = '9286'
store_id_input = '16'
forecast_days_input = 7

task_forecast = Task(
    description="Generate demand forecast for product 9286, store 16, 7 days.",
    expected_output="A string with the predicted demand.",
    agent=demand_forecaster
)

task_inventory_check = Task(
    description="Check inventory for product 9286, store 16.",
    expected_output="A JSON string with inventory status.",
    agent=inventory_manager,
    context=[task_forecast]
)

task_pricing = Task(
    description="Recommend pricing for product 9286, store 16.",
    expected_output="A JSON string with price and discount.",
    agent=pricing_strategist,
    context=[task_inventory_check, task_forecast]
)
print("Tasks defined.")

# --- Define the Crew ---
print("Defining the Crew...")
retail_crew = Crew(
    agents=[demand_forecaster, inventory_manager, pricing_strategist],
    tasks=[task_forecast, task_inventory_check, task_pricing],
    process=Process.sequential,
    verbose=True
)
print("Crew defined.")

print("\n--- Kicking off the Retail Crew ---")
results = {}
try:
    # Task 1: Demand Forecast
    print("\nExecuting Demand Forecast Task...")
    demand_tool = DemandForecastTool()
    demand_result = demand_tool._run(product_id_input, store_id_input, forecast_days_input)
    task_forecast.output = demand_result
    results["demand_forecast"] = demand_result

    # Task 2: Inventory Check
    print("\nExecuting Inventory Check Task...")
    inventory_tool = InventoryCheckTool()
    inventory_result = inventory_tool._run(product_id_input, store_id_input)
    task_inventory_check.output = inventory_result
    results["inventory_status"] = inventory_result

    # Task 3: Pricing Recommendation
    print("\nExecuting Pricing Recommendation Task...")
    pricing_tool = PricingRecommendationTool()
    pricing_result = pricing_tool._run(product_id_input, store_id_input)
    task_pricing.output = pricing_result
    results["pricing_recommendation"] = pricing_result

    # Create a user-friendly summary
    demand_value = json.loads(demand_result) if demand_result.startswith("{") else demand_result
    inventory_data = json.loads(inventory_result)
    pricing_data = json.loads(pricing_result)

    # Fetch current stock for display (run a quick query)
    conn = sqlite3.connect('data\\inventory.db')  # Adjust path
    cursor = conn.cursor()
    cursor.execute("SELECT stock_levels, reorder_point FROM inventory_levels WHERE product_id = ? AND store_id = ?", (product_id_input, store_id_input))
    stock_data = cursor.fetchone()
    current_stock = stock_data[0] if stock_data else "Unknown"
    reorder_point = stock_data[1] if stock_data else "Unknown"
    conn.close()

    summary_lines = [
        "Retail Analysis Summary for Product 9286 at Store 16:",
        f"- Expected Sales (Next 7 Days): {demand_value} units",
        "- Inventory Status:",
        f"  * Current Stock: {current_stock} units (Reorder when below {reorder_point} units)",
        f"  * Low Stock Items: {len(inventory_data['low_stock'])} (None)" if not inventory_data["low_stock"] else f"  * Low Stock Items: {len(inventory_data['low_stock'])}",
        f"  * Expiring Soon: {len(inventory_data['expiring_soon'])} (None)" if not inventory_data["expiring_soon"] else f"  * Expiring Soon: {len(inventory_data['expiring_soon'])}",
        "- Pricing Recommendation:",
        f"  * New Price: ${pricing_data['price']:.2f}",
        f"  * Discount: {pricing_data['discount_percentage']}% off"
    ]
    
    # If low_stock has items, add details
    if inventory_data["low_stock"]:
        summary_lines.insert(5, "    Details:")
        for item in inventory_data["low_stock"]:
            summary_lines.insert(6, f"    - Product {item['product_id']}, Store {item['store_id']}: {item['current_stock']} units (Reorder at {item['reorder_point']})")

    final_output = "\n".join(summary_lines)

    print("\n--- Crew Execution Finished ---")
    print("\nFinal Summary of All Tasks:")
    print(final_output)

except Exception as e:
    print(f"\n--- ERROR DURING CREW EXECUTION ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

# --- Cleanup ---
print("\n--- Cleaning up ---")
if 'demand_agent_logic' in locals() and demand_agent_logic: demand_agent_logic.close_connection()
if 'inventory_agent_logic' in locals() and inventory_agent_logic: inventory_agent_logic.close_connection()
if 'pricing_agent_logic' in locals() and pricing_agent_logic: pricing_agent_logic.close_connection()
print("Cleanup complete.")