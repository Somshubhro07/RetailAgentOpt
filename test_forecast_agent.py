# test_forecast_agent.py
from agents import DemandForecastAgent # Import your agent class
import db_utils # To potentially verify data if needed
import sqlite3

# --- CONFIGURATION: Set Product/Store to test ---
# <<< CHANGE THESE to valid IDs from your data >>>
TEST_PRODUCT_ID = '5406'
TEST_STORE_ID = '67'
FORECAST_DAYS = 7

if __name__ == "__main__":
    print("--- Starting Demand Forecast Agent Test ---")
    print(f"Testing with Product ID: {TEST_PRODUCT_ID}, Store ID: {TEST_STORE_ID}")

    # Verify test data exists in DB
    conn_check = db_utils.create_connection()
    if conn_check:
        history_check = db_utils.get_demand_forecast(conn_check, TEST_PRODUCT_ID, TEST_STORE_ID)
        if not history_check:
            print(f"WARNING: No historical data found for {TEST_PRODUCT_ID}/{TEST_STORE_ID}. Test may fail or yield poor results.")
        else:
            print(f"Found {len(history_check)} historical records for testing.")
        conn_check.close()
        print("DB connection check closed.")
    else:
        print("WARNING: Could not connect to DB to verify test data.")

    # Instantiate the agent
    forecast_agent = None # Initialize to None
    try:
        forecast_agent = DemandForecastAgent()
    except ConnectionError as e:
        print(f"Failed to initialize agent: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during agent initialization: {e}")
        exit()

    # Call the generate_forecast method
    # Ensure your Ollama service is running!
    print("\nCalling generate_forecast...")
    prediction_result, raw_ollama_response = forecast_agent.generate_forecast(
        product_id=TEST_PRODUCT_ID,
        store_id=TEST_STORE_ID,
        forecast_horizon_days=FORECAST_DAYS
    )

    print("\n--- Test Results ---")
    print(f"Raw Ollama Response:\n```\n{raw_ollama_response}\n```") # Print the raw response clearly

    if prediction_result is not None:
        print(f"\n>>> Parsed Prediction for next {FORECAST_DAYS} days: {prediction_result}")
    else:
        print("\n>>> Failed to parse a prediction or generate forecast.")

    # Close the agent's connection
    if forecast_agent:
        forecast_agent.close_connection()

    print("\n--- Test Finished ---")