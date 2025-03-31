# agents.py
import re
from crewai import Agent
import ollama # Import the ollama library
import db_utils # Import our database utility functions
import pandas as pd # Needed for potential data manipulation/summarization
import json # Useful for formatting data for LLM prompts

# --- Existing InventoryAgent Class (Keep As Is) ---
class InventoryAgent():
    # ... (keep the existing code for InventoryAgent here) ...
    def __init__(self):
        # Establish a database connection for this agent instance
        # Consider connection pooling if performance becomes an issue
        self.db_conn = db_utils.create_connection()
        if not self.db_conn:
            # Handle connection failure appropriately - maybe raise an exception
            print("FATAL: InventoryAgent failed to connect to the database.")
            # Depending on your main loop, you might want to exit or retry
            raise ConnectionError("Failed to connect to database for InventoryAgent")

        # Define the agent using CrewAI's Agent class
        # We'll add specific Ollama integration (llm=...) later if needed within CrewAI's structure
        self.agent = Agent(
            role='Inventory Management Specialist',
            goal='Monitor stock levels, check reorder points, identify expiring stock, and request restocking based on demand forecasts and current inventory.',
            backstory=(
                "An meticulous AI agent responsible for tracking every item across all stores and warehouses. "
                "It ensures shelves are optimally stocked by analyzing current levels against reorder points, "
                "considering supplier lead times and product expiry dates. It communicates stock status and needs "
                "proactively to prevent stockouts and minimize waste."
            ),
            verbose=True, # Set to True to see agent's thinking process
            allow_delegation=False # This agent likely performs its own tasks directly
            # tools=[...] # We can add specific tools later (e.g., database query tools)
            # llm=... # Specify Ollama model here if needed for internal reasoning later
        )
        print("InventoryAgent initialized.")

    # ... (keep the existing methods check_stock_status, update_inventory_after_sales, close_connection) ...
    def check_stock_status(self, product_id=None, store_id=None):
        """
        Checks current stock levels against reorder points and expiry dates.
        Returns a summary of low stock items and expiring items.
        """
        print(f"\nInventoryAgent: Checking stock status for product: {product_id or 'All'}, store: {store_id or 'All'}...")
        if not self.db_conn:
            print("Error: No database connection available.")
            return {"low_stock": [], "expiring_soon": []}

        try:
            inventory_data = db_utils.get_inventory_levels(self.db_conn, product_id, store_id)
            if not inventory_data:
                print("No inventory data found for the specified criteria.")
                return {"low_stock": [], "expiring_soon": []}

            low_stock_items = []
            expiring_items = []
            # current_date = pd.Timestamp.now().normalize() # Get today's date for expiry check

            for item in inventory_data:
                # Check against reorder point
                stock = item.get('stock_levels')
                reorder_point = item.get('reorder_point')
                pid = item.get('product_id')
                sid = item.get('store_id')

                if stock is not None and reorder_point is not None:
                    if stock < reorder_point:
                        print(f"  - LOW STOCK DETECTED: Product {pid}, Store {sid}. Stock: {stock}, Reorder Point: {reorder_point}")
                        low_stock_items.append({
                            'product_id': pid,
                            'store_id': sid,
                            'current_stock': stock,
                            'reorder_point': reorder_point
                        })

                # Check expiry date (Requires date conversion logic)
                # expiry_date_str = item.get('expiry_date')
                # if expiry_date_str:
                #     try:
                #         expiry_date = pd.to_datetime(expiry_date_str).normalize()
                #         # Define 'soon' (e.g., within 30 days)
                #         days_until_expiry = (expiry_date - current_date).days
                #         if 0 <= days_until_expiry <= 30: # Example threshold: expiring within 30 days
                #              print(f"  - EXPIRING SOON: Product {pid}, Store {sid}. Expires: {expiry_date_str} ({days_until_expiry} days)")
                #              expiring_items.append({
                #                  'product_id': pid,
                #                  'store_id': sid,
                #                  'expiry_date': expiry_date_str,
                #                  'days_left': days_until_expiry
                #              })
                #     except Exception as e:
                #          print(f"  - WARNING: Could not parse expiry date '{expiry_date_str}' for {pid}, {sid}. Error: {e}")


            print(f"Stock check complete. Low stock items: {len(low_stock_items)}, Expiring items: {len(expiring_items)}")
            return {
                "low_stock": low_stock_items,
                "expiring_soon": expiring_items # Add expiring items logic back if needed
            }

        except Exception as e:
            print(f"An error occurred during stock status check: {e}")
            return {"low_stock": [], "expiring_soon": []}

    def update_inventory_after_sales(self, sales_data):
        """
        Updates inventory levels based on sales data.
        'sales_data' should be a list of dictionaries, e.g.,
        [{'product_id': 'p1', 'store_id': 's1', 'quantity_sold': 5}, ...]
        """
        print("\nInventoryAgent: Updating inventory based on sales data...")
        if not self.db_conn:
            print("Error: No database connection available.")
            return False

        success_count = 0
        for sale in sales_data:
            pid = sale.get('product_id')
            sid = sale.get('store_id')
            qty_sold = sale.get('quantity_sold')

            if not all([pid, sid, qty_sold]):
                print(f"  - Skipping invalid sales record: {sale}")
                continue

            # 1. Get current stock
            current_inventory = db_utils.get_inventory_levels(self.db_conn, product_id=pid, store_id=sid)
            if not current_inventory:
                print(f"  - WARNING: No inventory record found for Product {pid}, Store {sid}. Cannot update stock.")
                continue

            # Assuming get_inventory_levels returns a list of dicts, take the first one
            inv_record = current_inventory[0]
            current_stock = inv_record.get('stock_levels')
            if current_stock is None:
                 print(f"  - WARNING: Stock level is null for Product {pid}, Store {sid}. Cannot update.")
                 continue

            # 2. Calculate new stock
            new_stock = current_stock - qty_sold
            if new_stock < 0:
                print(f"  - WARNING: Sale of {qty_sold} for Product {pid}, Store {sid} results in negative stock ({new_stock}). Setting stock to 0.")
                new_stock = 0

            # 3. Update database
            if db_utils.update_stock_level(self.db_conn, pid, sid, new_stock):
                success_count += 1
            else:
                 print(f"  - Failed to update stock for Product {pid}, Store {sid}.")

        print(f"Inventory update complete. {success_count}/{len(sales_data)} sales records processed.")
        return success_count == len(sales_data)


    def close_connection(self):
        """ Closes the agent's database connection. """
        if self.db_conn:
            self.db_conn.close()
            print("InventoryAgent DB connection closed.")


# --- NEW DemandForecastAgent Class ---
class DemandForecastAgent():
    def __init__(self):
        self.db_conn = db_utils.create_connection()
        if not self.db_conn:
            print("FATAL: DemandForecastAgent failed to connect to the database.")
            raise ConnectionError("Failed to connect to database for DemandForecastAgent")

        # Define CrewAI agent
        self.agent = Agent(
            role='Demand Forecasting Analyst',
            goal='Analyze historical sales data, seasonality, promotions, and other factors to accurately predict future product demand for specific stores.',
            backstory=(
                "An analytical AI agent specialized in predicting retail demand. It leverages historical data "
                "and understands influencing factors like price, promotions, and seasonality. It uses advanced "
                "analytical techniques, including insights from local LLMs, to provide reliable demand forecasts, "
                "helping optimize inventory levels."
            ),
            verbose=True,
            allow_delegation=False # Forecast generation is its core task
            # llm=... # Optionally specify Ollama model via CrewAI's config later
        )
        # Initialize Ollama client directly if needed for more control than CrewAI provides
        # self.ollama_client = ollama.Client() # Example
        print("DemandForecastAgent initialized.")

    def _prepare_data_for_llm(self, history_data, recent_records=30):
        """ Prepares a summary of historical data for the LLM prompt. """
        if not history_data:
            return "No historical data available."

        # Convert list of dicts to DataFrame for easier manipulation
        df = pd.DataFrame(history_data)

        # Select most recent records (e.g., last 30 entries based on date)
        # Assumes data is sorted by date descending from db_utils
        recent_df = df.head(recent_records)

        # Create a concise summary (example: JSON string)
        # Include key columns: date, sales_quantity, price, promotions, seasonality_factors
        # You might want to add more sophisticated summarization (e.g., weekly aggregates)
        summary_data = recent_df[[
            'date', 'sales_quantity', 'price', 'promotions', 'seasonality_factors', 'demand_trend'
        ]].to_dict(orient='records')

        # Convert to JSON string for embedding in the prompt
        return json.dumps(summary_data, indent=2)


    def generate_forecast(self, product_id, store_id, forecast_horizon_days=7):
        """
        Generates a demand forecast for a given product/store for a number of days ahead.
        """
        print(f"\nDemandForecastAgent: Generating forecast for Product {product_id}, Store {store_id} for next {forecast_horizon_days} days...")
        if not self.db_conn:
            print("Error: No database connection.")
            return None # Indicate failure

        # 1. Fetch relevant historical data
        # Fetch more history than just the forecast horizon for context
        # Adjust date range or limit as needed
        historical_data = db_utils.get_demand_forecast(self.db_conn, product_id=product_id, store_id=store_id)

        if not historical_data:
            print(f"  - No historical demand data found for Product {product_id}, Store {store_id}.")
            # Maybe return a default forecast or 0?
            return 0 # Example: return 0 if no history

        # 2. Prepare data summary for Ollama prompt
        data_summary = self._prepare_data_for_llm(historical_data)
        print(f"  - Prepared data summary for LLM:\n{data_summary[:500]}...") # Print start of summary

        # 3. Construct the prompt for Ollama (gemma:2b)
        # This requires careful prompt engineering!
        prompt = f"""
        You are a retail demand forecasting expert.
        Analyze the following recent historical sales data for Product ID '{product_id}' at Store ID '{store_id}':
        {data_summary}

        Based *only* on the trends, seasonality, price points, and promotions visible in this data,
        predict the total expected sales quantity for the next {forecast_horizon_days} days.

        Provide your prediction as a single integer number only, without any explanation or extra text.
        Prediction:
        """
        print(f"\n  - Sending prompt to Ollama (gemma:2b)...")
        # print(f"--- Prompt Start ---\n{prompt}\n--- Prompt End ---") # Uncomment to debug prompt

        # 4. Call Ollama
        try:
            response = ollama.chat(
                model='gemma:2b', # Use the desired model
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            print(f"  - Received raw response from Ollama: '{raw_response_content}'")

            # 5. Parse the response (extract the number) - DEBUGGING VERSION
            predicted_demand = None
            raw_response_content = raw_response_content.strip() # Ensure leading/trailing whitespace removed
            print(f"  - DEBUG: Attempting to parse: '{raw_response_content}'")
            try:
                # Attempt 1: Look for pattern like "is XXX" or "prediction: XXX"
                # Uses re.IGNORECASE for case-insensitivity
                match1 = re.search(r'\b(?:is|prediction:?)\s+(\d+)', raw_response_content, re.IGNORECASE)
                if match1:
                    predicted_demand = int(match1.group(1))
                    print(f"  - DEBUG: Matched Attempt 1 (is/prediction): {predicted_demand}")
                else:
                    print("  - DEBUG: Attempt 1 did not match.")
                    # Attempt 2: Look for a number possibly followed by 'units' etc. at the end
                    match2 = re.search(r'(\d+)\s*(?:units|items)?\.?\s*$', raw_response_content)
                    if match2:
                        predicted_demand = int(match2.group(1))
                        print(f"  - DEBUG: Matched Attempt 2 (end of string): {predicted_demand}")
                    else:
                        print("  - DEBUG: Attempt 2 did not match.")
                        # Attempt 3: Fallback - Find the LAST number in the string
                        all_numbers = re.findall(r'\d+', raw_response_content)
                        if all_numbers:
                            predicted_demand = int(all_numbers[-1])
                            print(f"  - DEBUG: Matched Attempt 3 (last number): {predicted_demand}")
                        else:
                            print(f"  - DEBUG: Attempt 3 found no numbers.")
                            predicted_demand = None # Explicitly set to None if no numbers found


                # Now handle the outcome
                if predicted_demand is not None:
                    print(f"  - Parsed predicted demand: {predicted_demand}")
                    # Return tuple: (prediction, raw_response)
                    return predicted_demand, raw_response_content
                else:
                    # Fallback strategy if no number parsed after all attempts
                    print(f"  - WARNING: No prediction number parsed. Using fallback.")
                    fallback_demand = None
                    if len(historical_data) >= 7:
                        fallback_demand = int(pd.DataFrame(historical_data).head(7)['sales_quantity'].mean())
                        print(f"  - Using fallback demand (avg of last 7): {fallback_demand}")
                    elif historical_data: # Fallback if less than 7 days history
                        fallback_demand = int(pd.DataFrame(historical_data)['sales_quantity'].mean())
                        print(f"  - Using fallback demand (avg of available {len(historical_data)} records): {fallback_demand}")
                    else: # Absolute fallback
                         fallback_demand = 0
                         print(f"  - Using absolute fallback demand: {fallback_demand}")
                    # Return tuple: (fallback_prediction, raw_response)
                    return fallback_demand, raw_response_content

            except ValueError as ve:
                print(f"  - WARNING: Could not convert parsed value to integer. Error: {ve}")
                # Return tuple: (None, raw_response)
                return None, raw_response_content
            # Make sure not to catch the main Ollama call exception here
            # except Exception as e: # This was too broad before
            #      print(f"  - ERROR during response parsing: {e}")
            #      return None, raw_response_content

        except Exception as e:
            print(f"  - ERROR: Failed to get prediction from Ollama: {e}")
            return None, f"Ollama Call Error: {e}" # Return None, and the error message # Indicate failure

    def close_connection(self):
        """ Closes the agent's database connection. """
        if self.db_conn:
            self.db_conn.close()
            print("DemandForecastAgent DB connection closed.")

# --- NEW PricingAgent Class ---
class PricingAgent():
    def __init__(self, base_margin_percent=15.0): # Example: Set a default minimum margin
        self.db_conn = db_utils.create_connection()
        if not self.db_conn:
            print("FATAL: PricingAgent failed to connect to the database.")
            raise ConnectionError("Failed to connect to database for PricingAgent")

        self.base_margin_percent = base_margin_percent

        # Define CrewAI agent
        self.agent = Agent(
            role='Retail Pricing Strategist',
            goal='Analyze product inventory levels, demand forecasts, competitor pricing, costs, and other factors to recommend optimal pricing and promotional discounts, ensuring minimum profitability.',
            backstory=(
                "A data-driven AI agent focused on maximizing profitability and sales velocity through smart pricing. "
                "It constantly monitors market dynamics, inventory health, and demand patterns using both predefined rules and LLM reasoning "
                "to identify opportunities for price adjustments, ensuring competitiveness while protecting margins."
            ),
            verbose=True,
            allow_delegation=False
            # llm=... # Ollama integration will be via direct calls for now
        )
        print("PricingAgent initialized.")

    def _get_required_data(self, product_id, store_id):
        """ Helper function to fetch all data needed for pricing decisions. """
        print(f"  PricingAgent: Fetching data for Product {product_id}, Store {store_id}...")
        if not self.db_conn:
            print("  - ERROR: No database connection in _get_required_data.")
            return None # Return None if connection failed earlier

        try:
            # Fetch data using db_utils (ensure IDs are passed correctly for INTEGER columns)
            product_id_int = int(product_id)
            store_id_int = int(store_id)

            pricing_info_list = db_utils.get_generic_query(self.db_conn, "pricing_data", {"product_id": product_id_int, "store_id": store_id_int})
            pricing_info = pricing_info_list[0] if pricing_info_list else {}

            inventory_info_list = db_utils.get_inventory_levels(self.db_conn, product_id_int, store_id_int)
            inventory_info = inventory_info_list[0] if inventory_info_list else {}

            demand_info_list = db_utils.get_demand_forecast(self.db_conn, product_id_int, store_id_int) # Gets latest
            demand_info = demand_info_list[0] if demand_info_list else {}

            # --- Start of Corrected Assignment Section ---

            # 1. Create the initial combined_data dictionary with all fetched values
            # THIS LINE MUST BE PRESENT AND EXECUTED BEFORE USING combined_data BELOW
            combined_data = {
                "product_id": product_id_int, # Use int versions
                "store_id": store_id_int,     # Use int versions
                # Cost calculation will happen AFTER this dict is created
                "current_price": pricing_info.get('price'),
                "competitor_price": pricing_info.get('competitor_prices'),
                "current_discount": pricing_info.get('discounts', 0.0), # Default to 0 discount
                "sales_volume": pricing_info.get('sales_volume'),
                "customer_reviews": pricing_info.get('customer_reviews'),
                "return_rate": pricing_info.get('return_rate_'),
                "storage_cost": pricing_info.get('storage_cost'), # Get storage cost here
                "elasticity_index": pricing_info.get('elasticity_index'),
                "current_stock": inventory_info.get('stock_levels'),
                "reorder_point": inventory_info.get('reorder_point'),
                "demand_trend": demand_info.get('demand_trend'),
                "last_sales_qty" : demand_info.get('sales_quantity'),
                "promotions_active": demand_info.get('promotions', 'No') == 'Yes', # Boolean flag
                "seasonality": demand_info.get('seasonality_factors')
                # cost_of_goods will be added below
            }

            # 2. Now calculate cost_of_goods using values from combined_data
            # **Placeholder for Cost:** Refine this logic if you have better cost data
            # Use .get() with default for safety
            current_price_for_cost = combined_data.get('current_price')
            cost_of_goods = current_price_for_cost * 0.7 if current_price_for_cost is not None else 10.0 # Example: Default cost if price missing

            # 3. Add storage_cost (if available) *after* combined_data exists
            storage_cost_val = combined_data.get('storage_cost') # Get value once
            if storage_cost_val is not None:
                 try:
                     cost_of_goods += float(storage_cost_val)
                 except (ValueError, TypeError):
                     print(f"  - WARNING: Could not add storage_cost '{storage_cost_val}' to cost_of_goods.")

            # 4. Add the calculated cost_of_goods back into the dictionary
            combined_data['cost_of_goods'] = cost_of_goods

            # --- End of Corrected Assignment Section ---

            print(f"  PricingAgent: Combined data fetched (sample): cost_of_goods={combined_data['cost_of_goods']:.2f}, current_price={combined_data['current_price']}, current_stock={combined_data['current_stock']}")
            return combined_data

        except (ValueError, TypeError) as e:
            print(f"  - ERROR: Invalid ID format provided? Could not convert '{product_id}' or '{store_id}' to integer. Error: {e}")
            return None
        except Exception as e:
            print(f"  - ERROR: An unexpected error occurred in _get_required_data: {e}")
            # Optionally re-raise the exception if it shouldn't be silenced
            # raise e
            return None

    def _apply_pricing_rules(self, data):
        """ Applies predefined rules to suggest a price/discount. """
        print("  PricingAgent: Applying pricing rules...")
        reason = "Maintain current price (default)."
        suggested_price = data.get('current_price')
        suggested_discount = data.get('current_discount', 0.0)

        # --- Rule Execution (Apply in order of priority or combine effects) ---

        # 0. Basic Validation & Margin Check
        if suggested_price is None:
             return {"new_price": None, "discount_percent": 0.0, "reason": "Missing current price."}
        min_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
        print(f"    - Rule Check: Min Price (Cost + {self.base_margin_percent}% margin) = {min_price:.2f}")

        # 1. Competitor Matching Rule (Example: Undercut slightly if higher)
        if data.get('competitor_price') is not None and suggested_price > data['competitor_price']:
            potential_price = data['competitor_price'] * 0.99 # Undercut by 1%
            if potential_price >= min_price:
                suggested_price = round(potential_price, 2)
                suggested_discount = 0.0 # Reset discount if matching competitor
                reason = f"Price adjusted to slightly undercut competitor ({data['competitor_price']:.2f})."
                print(f"    - Rule Applied: Competitor Undercut. New Price: {suggested_price}")
            else:
                 print(f"    - Rule Skipped: Undercutting competitor would violate minimum margin.")


        # 2. Inventory Level Rule (Example: Discount high stock, especially if demand low/stable)
        stock = data.get('current_stock')
        reorder = data.get('reorder_point')
        trend = data.get('demand_trend')
        if stock is not None and reorder is not None and stock > reorder * 1.5: # Stock > 1.5x reorder point
             if trend in ['Decreasing', 'Stable']:
                 discount_increase = 5.0 # Add 5% discount
                 potential_discount = suggested_discount + discount_increase
                 potential_price = round(data['current_price'] * (1 - potential_discount / 100.0), 2)

                 if potential_price >= min_price:
                      suggested_discount = potential_discount
                      reason = f"High stock ({stock}) with {trend} demand. Increased discount to {suggested_discount}%."
                      print(f"    - Rule Applied: High Stock Discount. New Discount: {suggested_discount}%")
                 else:
                      print(f"    - Rule Skipped: High stock discount would violate minimum margin.")

        # 3. Promotion Rule (Example: Apply standard promo discount if active)
        # Note: This might override other rules depending on strategy
        if data.get('promotions_active', False):
             promo_discount = 15.0 # Example standard promo discount
             potential_price = round(data['current_price'] * (1 - promo_discount / 100.0), 2)
             if potential_price >= min_price:
                  # Decide: does promo override or add to existing discount? Let's override for simplicity.
                  suggested_discount = promo_discount
                  reason = f"Applying standard {promo_discount}% promotion."
                  print(f"    - Rule Applied: Promotion Active. New Discount: {suggested_discount}%")
             else:
                  print(f"    - Rule Skipped: Promotion discount would violate minimum margin.")


        # Recalculate final price based on final discount
        final_price = round(data['current_price'] * (1 - suggested_discount / 100.0), 2)
        # Ensure final price respects minimum margin absolute floor
        if final_price < min_price:
             final_price = min_price
             # Recalculate discount based on clamped price (optional)
             if data['current_price'] > 0:
                suggested_discount = round((1 - final_price / data['current_price']) * 100.0, 1)
             else:
                suggested_discount = 0.0
             reason += f" Adjusted to meet minimum margin floor."
             print(f"    - Final price adjusted to minimum margin: {final_price}. Effective discount: {suggested_discount}%")


        print(f"  PricingAgent: Rules finished. Reason: {reason}")
        return {"new_price": final_price, "discount_percent": suggested_discount, "reason": reason}


    def _get_ollama_price_recommendation(self, data, rule_recommendation):
        """ Sends pricing context AND rule suggestion to Ollama and parses recommendation. """
        print("  PricingAgent: Preparing data and prompt for Ollama validation/refinement...")

        # Format rule recommendation for prompt
        rule_reason = rule_recommendation.get('reason', 'N/A')
        rule_price = rule_recommendation.get('new_price')
        rule_discount = rule_recommendation.get('discount_percent')
        rule_summary = f"Our rule-based system analyzed the data and suggested: Action=[Set Price to {rule_price} / Discount to {rule_discount}%], Reason=[{rule_reason}]."

        prompt = f"""
        You are a Retail Pricing Strategist AI providing a second opinion.
        Analyze the following data for Product ID '{data['product_id']}' at Store ID '{data['store_id']}':
        - Cost of Goods (Estimated): {data.get('cost_of_goods', 'N/A'):.2f}
        - Current Price: {data.get('current_price', 'N/A')}
        - Current Stock Level: {data.get('current_stock', 'N/A')} (Reorder Point: {data.get('reorder_point', 'N/A')})
        - Demand Trend: {data.get('demand_trend', 'N/A')}
        - Last Sales Quantity: {data.get('last_sales_qty', 'N/A')}
        - Competitor Price: {data.get('competitor_price', 'N/A')}
        - Elasticity Index: {data.get('elasticity_index', 'N/A')} (Higher means more price sensitive)
        - Storage Cost: {data.get('storage_cost', 'N/A')}
        - Promotion Active: {data.get('promotions_active', False)}
        - Seasonality: {data.get('seasonality', 'N/A')}

        Context: {rule_summary}

        Based on *all* factors, especially considering elasticity and potential interactions the rules might miss:
        1. Briefly state if you agree with the rule-based suggestion.
        2. Provide your final recommendation as a JSON object like this (only the JSON):
        {{
          "llm_reason": "Your brief reasoning focusing on factors rules might miss.",
          "llm_agrees_with_rule": <true_or_false>,
          "llm_suggested_price": <price_number_or_null_if_discount>,
          "llm_suggested_discount_percent": <discount_percentage_or_null_if_price_set>
        }}
        """
        print("  - Sending enhanced prompt to Ollama (gemma:2b)...")

        try:
            response = ollama.chat(
                model='gemma:2b', # Or another suitable model
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            print(f"  - Received raw response from Ollama: {raw_response_content}")

            # Parse the JSON response
            try:
                # Find JSON block (robust against leading/trailing text)
                json_match = re.search(r'\{.*\}', raw_response_content, re.DOTALL)
                if json_match:
                    json_string = json_match.group(0)
                    recommendation = json.loads(json_string)
                    # Basic validation
                    if 'llm_reason' in recommendation and 'llm_agrees_with_rule' in recommendation:
                         print(f"  - Parsed Ollama recommendation: {recommendation}")
                         return recommendation
                    else:
                         print("  - WARNING: Parsed JSON lacks required keys.")
                         return {"llm_reason": "Failed to parse required keys.", "llm_agrees_with_rule": None}
                else:
                     print("  - WARNING: No JSON object found in Ollama response.")
                     return {"llm_reason": "No JSON found in response.", "llm_agrees_with_rule": None}

            except json.JSONDecodeError as jde:
                print(f"  - WARNING: Could not parse JSON response from Ollama: {jde}")
                return {"llm_reason": f"JSON parsing error: {jde}", "llm_agrees_with_rule": None}
            except Exception as e:
                 print(f"  - ERROR during JSON parsing: {e}")
                 return {"llm_reason": f"Parsing exception: {e}", "llm_agrees_with_rule": None}

        except Exception as e:
            print(f"  - ERROR: Failed to get recommendation from Ollama: {e}")
            return {"llm_reason": f"Ollama call failed: {e}", "llm_agrees_with_rule": None}


    def recommend_price(self, product_id, store_id):
        """
        Recommends a new price or discount using a hybrid rule-based + LLM approach.
        Returns a dictionary with final decision and reasoning.
        """
        print(f"\nPricingAgent: Generating HYBRID price recommendation for Product {product_id}, Store {store_id}...")

        # 1. Fetch data
        data = self._get_required_data(product_id, store_id)
        if not data or data.get('current_price') is None:
             print("  - ERROR: Insufficient data.")
             # Return error structure
             return {"recommendation": "Error: Insufficient data", "final_price": None, "final_discount": None, "rule_reason": None, "llm_reason": None}

        # 2. Apply Rules
        rule_recommendation = self._apply_pricing_rules(data)

        # 3. Get LLM Validation/Refinement
        # Pass both the original data and the rule recommendation to Ollama
        ollama_opinion = self._get_ollama_price_recommendation(data, rule_recommendation)

        # 4. Reconciliation Logic (Example: Use rule price, add LLM reasoning)
        # More complex logic could be added here to override rules based on LLM confidence/reasoning
        final_price = rule_recommendation.get('new_price')
        final_discount = rule_recommendation.get('discount_percent')
        combined_reason = f"Rule Action: [{rule_recommendation.get('reason', 'N/A')}]. LLM Analysis: [{ollama_opinion.get('llm_reason', 'N/A')}]"

        # Simple override example (use LLM if it strongly disagrees and provides values)
        # if ollama_opinion.get('llm_agrees_with_rule') is False:
        #     llm_price = ollama_opinion.get('llm_suggested_price')
        #     llm_discount = ollama_opinion.get('llm_suggested_discount_percent')
        #     if llm_price is not None:
        #         # Basic check: Ensure LLM price respects minimum margin
        #         min_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
        #         if llm_price >= min_price:
        #             print("  - Reconciliation: Overriding rule with LLM suggested price.")
        #             final_price = llm_price
        #             final_discount = 0.0
        #             combined_reason += " [Decision: Used LLM price override]"
        #         else:
        #              print("  - Reconciliation: LLM suggested price below margin, keeping rule price.")
        #              combined_reason += " [Decision: Kept rule price - LLM suggestion below margin]"
        #     elif llm_discount is not None:
        #          # Apply similar logic for discount, ensuring price doesn't go below margin
        #          print("  - Reconciliation: Overriding rule with LLM suggested discount.")
        #          final_discount = llm_discount
        #          final_price = round(data['current_price'] * (1 - final_discount / 100.0), 2)
        #          # Re-check margin
        #          min_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
        #          if final_price < min_price:
        #              # Handle below margin case if LLM discount causes it
        #              final_price = min_price
        #              # ... recalculate discount ...
        #              print("  - Reconciliation: LLM discount adjusted for minimum margin.")
        #          combined_reason += " [Decision: Used LLM discount override]"


        print(f"  - Pricing Recommendation: {combined_reason}")
        print(f"  - Final Suggested Price: {final_price}, Final Suggested Discount: {final_discount}%")

        return {
            "recommendation": combined_reason,
            "final_price": final_price,
            "final_discount": final_discount,
            "rule_reason": rule_recommendation.get('reason'), # Keep separate reasons if needed
            "llm_reason": ollama_opinion.get('llm_reason')
        }


    def close_connection(self):
        """ Closes the agent's database connection. """
        if self.db_conn:
            self.db_conn.close()
            print("PricingAgent DB connection closed.")



# --- Add SupplierAgent, CoordinatorAgent placeholders later ---