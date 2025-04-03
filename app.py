# app.py
import sqlite3
from flask import Flask, render_template, request, jsonify
from agents import DemandForecastAgent, InventoryAgent, PricingAgent
from langchain_ollama import OllamaLLM
from crewai import Agent as CrewAgent, Task, Crew, Process

app = Flask(__name__)

# Configure Ollama LLM
ollama_llm = OllamaLLM(model="ollama/gemma:2b", base_url="http://localhost:11434")
print(f"Ollama LLM configured with model '{ollama_llm.model}' at base URL '{ollama_llm.base_url}'.")

# Initialize agents
demand_agent = DemandForecastAgent()
inventory_agent = InventoryAgent()
pricing_agent = PricingAgent(base_margin_percent=15.0)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to process input and get agent outputs
@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.json
        product_id = data.get('product_id')
        store_id = data.get('store_id')
        forecast_horizon_days = int(data.get('forecast_horizon_days', 7))

        # Get agent outputs
        demand_forecast, _ = demand_agent.generate_forecast(product_id, store_id, forecast_horizon_days)
        inventory_status = inventory_agent.check_inventory(product_id, store_id, demand_forecast)
        pricing_recommendation = pricing_agent.recommend_price(product_id, store_id)

        # Prepare response
        response = {
            'demand_forecast': f"Expected sales: {demand_forecast:.2f} units over {forecast_horizon_days} days",
            'inventory_status': inventory_status['message'],
            'pricing_recommendation': pricing_recommendation['recommendation'],
            'final_price': f"₹{pricing_recommendation['final_price']:.2f}",
            'final_discount': f"{pricing_recommendation['final_discount']:.2f}%"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)