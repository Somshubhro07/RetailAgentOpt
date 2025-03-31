# test_pricing_agent.py
from agents import PricingAgent # Import your agent class
import db_utils # To potentially verify data if needed
import sqlite3

# --- CONFIGURATION: Set Product/Store to test ---
# <<< CHANGE THESE to valid IDs from your data >>>
# Make sure this Product/Store combo has data in ALL THREE tables:
# pricing_data, inventory_levels, demand_forecasts
TEST_PRODUCT_ID = '9286'
TEST_STORE_ID = '16'

if __name__ == "__main__":
    print("--- Starting Pricing Agent Test ---")
    print(f"Testing with Product ID: {TEST_PRODUCT_ID}, Store ID: {TEST_STORE_ID}")

    # Optional: Add checks using db_utils to ensure data exists for this combo
    # in pricing_data, inventory_levels, and demand_forecasts tables.
    print("Verifying data prerequisites...")
    conn_check = db_utils.create_connection()
    data_ok = True
    if conn_check:
        pricing_check = db_utils.get_generic_query(conn_check, "pricing_data", {"product_id": TEST_PRODUCT_ID, "store_id": TEST_STORE_ID})
        inventory_check = db_utils.get_inventory_levels(conn_check, TEST_PRODUCT_ID, TEST_STORE_ID)
        demand_check = db_utils.get_demand_forecast(conn_check, TEST_PRODUCT_ID, TEST_STORE_ID)
        if not pricing_check:
            print(f"WARNING: No data found in pricing_data for {TEST_PRODUCT_ID}/{TEST_STORE_ID}.")
            data_ok = False
        if not inventory_check:
            print(f"WARNING: No data found in inventory_levels for {TEST_PRODUCT_ID}/{TEST_STORE_ID}.")
            data_ok = False
        if not demand_check:
            print(f"WARNING: No data found in demand_forecasts for {TEST_PRODUCT_ID}/{TEST_STORE_ID}.")
            data_ok = False
        if data_ok:
            print("Found data in relevant tables.")
        conn_check.close()
        print("DB connection check closed.")
    else:
        print("WARNING: Could not connect to DB to verify test data.")
        data_ok = False

    if not data_ok:
        print("\n--- Test Aborted: Missing required data for test IDs ---")
        exit() # Stop if data is missing

    # Instantiate the agent
    pricing_agent = None # Initialize to None
    try:
        # You can adjust base_margin_percent if needed for testing
        pricing_agent = PricingAgent(base_margin_percent=15.0)
    except ConnectionError as e:
        print(f"Failed to initialize agent: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during agent initialization: {e}")
        exit()

    # Call the recommend_price method
    # Ensure your Ollama service is running!
    print("\nCalling recommend_price...")
    recommendation_output = pricing_agent.recommend_price(
        product_id=TEST_PRODUCT_ID,
        store_id=TEST_STORE_ID
    )

    print("\n--- Test Results ---")
    print("Full Recommendation Output:")
    # Pretty print the dictionary
    import json
    print(json.dumps(recommendation_output, indent=4))

    print("\n--- Key Results ---")
    print(f"Rule Reason: {recommendation_output.get('rule_reason')}")
    print(f"LLM Reason: {recommendation_output.get('llm_reason')}")
    print(f"Final Recommendation: {recommendation_output.get('recommendation')}")
    print(f"Final Price: {recommendation_output.get('final_price')}")
    print(f"Final Discount: {recommendation_output.get('final_discount')}%")


    # Close the agent's connection
    if pricing_agent:
        pricing_agent.close_connection()

    print("\n--- Test Finished ---")