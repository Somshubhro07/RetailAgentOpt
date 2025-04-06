# app.py
import sqlite3
from flask import Flask, render_template, request, jsonify
from agents import DemandForecastAgent, InventoryAgent, PricingAgent
import db_utils
import os
import io
import sys
import time

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize agents
demand_agent = DemandForecastAgent()
inventory_agent = InventoryAgent()
pricing_agent = PricingAgent(base_margin_percent=15.0)

# Ensure data directory exists
if not os.path.exists(db_utils.DATA_DIR):
    os.makedirs(db_utils.DATA_DIR)

# Initialize database with sample data on first run
conn = db_utils.create_connection()
if conn and not db_utils.get_inventory_levels(conn) and not db_utils.get_demand_forecast(conn) and not db_utils.get_generic_query(conn, "pricing_data"):
    db_utils.init_database(conn)
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.json
        product_id = data.get('product_id')
        store_id = data.get('store_id')
        forecast_horizon_days = int(data.get('forecast_horizon_days', 7))

        # Create a new database connection for this request
        conn = db_utils.create_connection()
        if not conn:
            return jsonify({'error': 'Failed to connect to database'}), 500

        # Capture console-like output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # Simulate processing steps with delays for loading messages
        loading_steps = [
            "Forecasting demand...",
            "Getting the correct price and discount...",
            "Making the best prompts for you to get the perfect output..."
        ]
        time.sleep(7)  # Simulate initial delay

        # Get agent outputs
        demand_forecast, ollama_response = demand_agent.generate_forecast(product_id, store_id, forecast_horizon_days, conn)
        time.sleep(5)  # Simulate processing
        inventory_status = inventory_agent.calculate_order_request(product_id, store_id, demand_forecast or 0, conn)
        time.sleep(5)  # Simulate processing
        pricing_recommendation = pricing_agent.recommend_price(product_id, store_id, conn)

        # Restore stdout and get the captured output
        sys.stdout = old_stdout
        thinking_log = buffer.getvalue().split('\n')

        conn.close()

        # Prepare detailed forecast data for the graph
        forecast_data = [demand_forecast / forecast_horizon_days if demand_forecast else 0] * forecast_horizon_days
        # Simulate some variation for a more interesting graph
        forecast_data = [x * (1 + (i * 0.1)) for i, x in enumerate(forecast_data)]

        # Prepare response
        response = {
            'demand_forecast': f"Expected sales: {demand_forecast or 0:.2f} units over {forecast_horizon_days} days" if demand_forecast is not None else "Error: Demand forecast failed.",
            'inventory_status': inventory_status['reason'],
            'order_quantity': inventory_status['order_quantity'],
            'pricing_recommendation': pricing_recommendation['recommendation'],
            'final_price': f"₹{pricing_recommendation['final_price']:.2f}" if pricing_recommendation['final_price'] else "N/A",
            'final_discount': f"{pricing_recommendation['final_discount']:.2f}%" if pricing_recommendation['final_discount'] else "N/A",
            'forecast_data': forecast_data,  # Detailed data for the graph
            'thinking_log': [line.strip() for line in thinking_log if line.strip()],
            'loading_steps': loading_steps  # Pass loading messages to frontend
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)