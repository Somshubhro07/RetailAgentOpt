# agents.py
import re
import json
import datetime
import sqlite3 # Needed for expiry date check
import pandas as pd
import ollama # Make sure ollama is imported

# Assuming db_utils provides necessary functions like:
# create_connection, get_demand_forecast, get_inventory_levels,
# get_generic_query(table_name, filters), update_stock_level
import db_utils

# --- Base Agent Logic Class (Optional but good practice) ---
# Helps manage DB connection consistently
class BaseAgentLogic:
    def __init__(self, agent_name="BaseAgent"):
        self.agent_name = agent_name
        self.db_conn = db_utils.create_connection()
        if not self.db_conn:
            print(f"FATAL: {self.agent_name} failed to connect to the database.")
            # Depending on main loop, might want to exit or retry
            raise ConnectionError(f"Failed to connect to database for {self.agent_name}")
        print(f"{self.agent_name} initialized with DB connection.")

    def close_connection(self):
        """ Closes the agent's database connection. """
        if self.db_conn:
            self.db_conn.close()
            print(f"{self.agent_name} DB connection closed.")

    def __del__(self):
        # Ensure connection is closed when object is garbage collected
        self.close_connection()

# --- InventoryAgent Class ---
class InventoryAgent(BaseAgentLogic):
    def __init__(self):
        super().__init__(agent_name="InventoryAgent")
        # CrewAI Agent definition placeholder (actual definition is in main.py)
        self.role = 'Inventory Management Specialist'
        self.goal = 'Monitor stock levels, check reorder points, identify expiring stock, and calculate restocking needs based on demand forecasts and current inventory.'
        self.backstory = "Meticulous AI ensuring optimal stock levels by tracking items, analyzing stock against reorder points, considering lead times and expiry, and proactively managing inventory flow."

    def check_stock_status(self, product_id: str, store_id: str, expiry_threshold_days: int = 30) -> dict:
        """
        Checks stock levels vs reorder points and identifies items nearing expiry date.
        Returns a dictionary with 'low_stock' and 'expiring_soon' lists.
        """
        print(f"\nInventoryAgent: Checking stock status for product: {product_id}, store: {store_id}...")
        results = {"low_stock": [], "expiring_soon": []}
        if not self.db_conn:
            print("Error: No database connection.")
            return results # Return empty results on error

        try:
            # Ensure IDs are integers for querying
            product_id_int = int(product_id)
            store_id_int = int(store_id)
            inventory_data = db_utils.get_inventory_levels(self.db_conn, product_id_int, store_id_int)

            if not inventory_data:
                print("  - No inventory data found for the specified criteria.")
                return results

            current_date = pd.Timestamp.now().normalize() # Get today's date (without time)

            for item in inventory_data:
                stock = item.get('stock_levels')
                reorder_point = item.get('reorder_point')
                pid = item.get('product_id') # Already int from query
                sid = item.get('store_id')   # Already int from query

                # Check against reorder point
                if stock is not None and reorder_point is not None and isinstance(stock, (int, float)) and isinstance(reorder_point, (int, float)):
                    if stock < reorder_point:
                        print(f"  - LOW STOCK DETECTED: Product {pid}, Store {sid}. Stock: {stock}, Reorder Point: {reorder_point}")
                        results["low_stock"].append({
                            'product_id': pid, 'store_id': sid,
                            'current_stock': stock, 'reorder_point': reorder_point
                        })
                else:
                     print(f"  - WARNING: Missing or invalid stock ({stock}) or reorder_point ({reorder_point}) for {pid}/{sid}.")


                # Check expiry date (Implement the logic)
                expiry_date_str = item.get('expiry_date') # Stored as TEXT e.g., 'YYYY-MM-DD HH:MM:SS'
                if expiry_date_str:
                    try:
                        # Convert string to datetime object, normalize to remove time part
                        expiry_date = pd.to_datetime(expiry_date_str).normalize()
                        days_until_expiry = (expiry_date - current_date).days

                        if days_until_expiry < 0:
                             print(f"  - WARNING: Product {pid}, Store {sid} expired on {expiry_date_str}!")
                             # Optionally add to a separate 'expired' list
                        elif 0 <= days_until_expiry <= expiry_threshold_days:
                            print(f"  - EXPIRING SOON: Product {pid}, Store {sid}. Expires: {expiry_date_str} ({days_until_expiry} days)")
                            results["expiring_soon"].append({
                                'product_id': pid, 'store_id': sid,
                                'expiry_date': expiry_date_str, 'days_left': days_until_expiry
                            })
                    except Exception as e:
                         print(f"  - WARNING: Could not parse expiry date '{expiry_date_str}' for {pid}/{sid}. Error: {e}")

            print(f"Stock check complete. Low stock items: {len(results['low_stock'])}, Expiring items: {len(results['expiring_soon'])}")
            return results

        except Exception as e:
            print(f"An error occurred during stock status check: {e}")
            return {"low_stock": [], "expiring_soon": [], "error": str(e)} # Include error in result

    def calculate_order_request(self, product_id: str, store_id: str, demand_forecast: int) -> dict:
        """
        Calculates the suggested order quantity based on forecast, current stock,
        reorder point, lead time, and capacity.
        """
        print(f"\nInventoryAgent: Calculating order request for Product {product_id}, Store {store_id} with forecast {demand_forecast}...")
        order_request = {"product_id": product_id, "store_id": store_id, "order_quantity": 0, "reason": "No order needed."}
        if not self.db_conn:
             print("Error: No database connection.")
             order_request["reason"] = "Error: DB connection failed."
             return order_request

        try:
            product_id_int = int(product_id)
            store_id_int = int(store_id)
            inventory_data = db_utils.get_inventory_levels(self.db_conn, product_id_int, store_id_int)
            if not inventory_data:
                 order_request["reason"] = "Error: No inventory data found."
                 return order_request

            item = inventory_data[0] # Assume one item record per product/store
            stock = item.get('stock_levels')
            reorder_point = item.get('reorder_point')
            lead_time = item.get('supplier_lead_time_days')
            capacity = item.get('warehouse_capacity') # Or maybe store capacity? Using warehouse for now.

            if None in [stock, reorder_point, lead_time, capacity]:
                 order_request["reason"] = "Error: Missing critical inventory data (stock, reorder point, lead time, capacity)."
                 return order_request

            # Simple Order-Up-To Logic Example (Needs refinement based on strategy)
            # Target Stock = Demand during lead time + safety stock (e.g., reorder point)
            # Assuming forecast is total for horizon, estimate daily demand
            # THIS IS VERY SIMPLISTIC - replace with better inventory policy logic
            daily_demand_estimate = demand_forecast / 7 # Rough estimate from weekly forecast
            demand_during_lead_time = daily_demand_estimate * lead_time
            safety_stock = reorder_point # Using reorder point as proxy for safety stock needed
            target_stock_level = demand_during_lead_time + safety_stock

            needed_quantity = target_stock_level - stock
            suggested_order = 0

            if needed_quantity > 0:
                suggested_order = round(needed_quantity) # Round to nearest whole unit
                reason = f"Stock ({stock}) below target ({target_stock_level:.0f}). Need ~{needed_quantity:.0f} units."

                # Consider capacity constraint
                available_capacity = capacity - stock
                if suggested_order > available_capacity:
                     print(f"  - WARNING: Suggested order ({suggested_order}) exceeds available capacity ({available_capacity}). Clamping order.")
                     suggested_order = available_capacity
                     reason += f" Order clamped to available capacity ({available_capacity})."

                if suggested_order > 0 :
                     order_request["order_quantity"] = suggested_order
                     order_request["reason"] = reason
                     print(f"  - Order Request: {suggested_order} units for {product_id}/{store_id}. Reason: {reason}")
                else:
                     order_request["reason"] = f"Stock ({stock}) sufficient or capacity full. Target: {target_stock_level:.0f}, Available Capacity: {available_capacity}."

            else:
                 order_request["reason"] = f"Stock ({stock}) is at or above target level ({target_stock_level:.0f})."

            return order_request

        except Exception as e:
            print(f"An error occurred during order request calculation: {e}")
            order_request["reason"] = f"Error: {e}"
            return order_request

    def update_inventory_after_sales(self, sales_data_list: list):
        """
        Updates inventory levels based on a list of sales data dicts.
        Example sales_data_list: [{'product_id': '9286', 'store_id': '16', 'quantity_sold': 5}, ...]
        """
        print(f"\nInventoryAgent: Updating inventory for {len(sales_data_list)} sales records...")
        if not self.db_conn:
            print("Error: No database connection.")
            return False

        success_count = 0
        processed_count = 0
        for sale in sales_data_list:
            processed_count += 1
            try:
                pid = sale.get('product_id')
                sid = sale.get('store_id')
                qty_sold = sale.get('quantity_sold')

                if not all([pid, sid, qty_sold is not None]): # Allow qty_sold=0
                    print(f"  - Skipping invalid sales record (missing keys): {sale}")
                    continue

                # Ensure IDs are int for DB query/update
                pid_int = int(pid)
                sid_int = int(sid)
                qty_sold_num = int(qty_sold) # Assuming whole units sold

                # 1. Get current stock
                current_inventory = db_utils.get_inventory_levels(self.db_conn, pid_int, sid_int)
                if not current_inventory:
                    print(f"  - WARNING: No inventory record found for Product {pid_int}, Store {sid_int}. Cannot update stock.")
                    continue

                inv_record = current_inventory[0]
                current_stock = inv_record.get('stock_levels')
                if current_stock is None:
                     print(f"  - WARNING: Stock level is null for Product {pid_int}, Store {sid_int}. Cannot update.")
                     continue

                # 2. Calculate new stock
                new_stock = current_stock - qty_sold_num
                if new_stock < 0:
                    print(f"  - WARNING: Sale of {qty_sold_num} for Product {pid_int}, Store {sid_int} results in negative stock ({new_stock}). Setting stock to 0.")
                    new_stock = 0

                # 3. Update database
                if db_utils.update_stock_level(self.db_conn, pid_int, sid_int, new_stock):
                    success_count += 1
                else:
                     print(f"  - Failed to update stock for Product {pid_int}, Store {sid_int}.")

            except (ValueError, TypeError) as e:
                 print(f"  - Skipping invalid sales record (conversion error): {sale}. Error: {e}")
                 continue
            except Exception as e:
                 print(f"  - ERROR processing sales record {sale}: {e}")
                 continue # Skip to next record on unexpected error

        print(f"Inventory update complete. {success_count}/{processed_count} sales records successfully processed.")
        # Return True if all records were processed without DB update failures (ignoring skips)
        return success_count == (processed_count - (processed_count - success_count))


# --- DemandForecastAgent Class ---
class DemandForecastAgent(BaseAgentLogic):
    def __init__(self):
        super().__init__(agent_name="DemandForecastAgent")
        self.role = 'Demand Forecasting Analyst'
        self.goal = 'Analyze historical sales data and context to accurately predict future product demand.'
        self.backstory = "Analytical AI specialized in retail demand prediction, leveraging historical data and LLM insights."

    def _calculate_features(self, df):
        """ Calculates simple features from historical data """
        features = {}
        if df.empty: return features

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)

        # Ensure sales_quantity is numeric, coerce errors to NaN
        df['sales_quantity'] = pd.to_numeric(df['sales_quantity'], errors='coerce')
        df.dropna(subset=['sales_quantity'], inplace=True) # Remove rows where conversion failed
        df['sales_quantity'] = df['sales_quantity'].astype(int) # Convert to int if possible

        if df.empty: return features # Return if no valid numeric sales data

        if len(df) >= 1:
             features['latest_sales'] = int(df['sales_quantity'].iloc[0])
             features['latest_promo'] = df['promotions'].iloc[0]
             features['latest_seasonality'] = df['seasonality_factors'].iloc[0]
             features['latest_trend'] = df['demand_trend'].iloc[0]

        if len(df) >= 7:
             features['avg_sales_last_7'] = round(df['sales_quantity'].head(7).mean(), 1)
        else:
             features['avg_sales_last_N'] = round(df['sales_quantity'].mean(), 1) # Avg of available

        if len(df) >= 2:
             # Simple trend check (compare latest vs previous)
             if df['sales_quantity'].iloc[0] > df['sales_quantity'].iloc[1]:
                  features['short_term_trend'] = 'Increasing'
             elif df['sales_quantity'].iloc[0] < df['sales_quantity'].iloc[1]:
                  features['short_term_trend'] = 'Decreasing'
             else:
                  features['short_term_trend'] = 'Stable'
        return features

    def _prepare_data_for_llm(self, history_data, recent_records_for_context=10):
        """ Prepares a more informative summary for the LLM prompt. """
        if not history_data:
            return "No historical data available.", {}

        df = pd.DataFrame(history_data)
        if df.empty:
             return "No historical data available.", {}

        # Calculate features
        features = self._calculate_features(df.copy()) # Use copy to avoid modifying original list dicts

        # Provide recent raw data + calculated features
        recent_df = df.head(recent_records_for_context)
        summary_context = recent_df[[
            'date', 'sales_quantity', 'price', 'promotions', 'seasonality_factors', 'demand_trend'
        ]].to_string(index=False) # Use string format for potentially better LLM reading

        # Combine features and recent data context
        context_str = "Recent Data Points (Latest First):\n" + summary_context
        if features:
             context_str += "\n\nKey Features:\n"
             for key, value in features.items():
                  context_str += f"- {key.replace('_', ' ').title()}: {value}\n"

        return context_str, features

    def generate_forecast(self, product_id: str, store_id: str, forecast_horizon_days: int = 7):
        """ Generates demand forecast using DB data and Ollama """
        print(f"\nDemandForecastAgent: Generating forecast for Product {product_id}, Store {store_id} for next {forecast_horizon_days} days...")
        raw_response_for_return = "No response generated."
        if not self.db_conn:
            print("Error: No database connection.")
            return None, "Error: No DB connection."

        try:
            product_id_int = int(product_id)
            store_id_int = int(store_id)
            # Fetch maybe last 90 days for better context? Adjust limit as needed.
            historical_data = db_utils.get_demand_forecast(self.db_conn, product_id=product_id_int, store_id=store_id_int) # get_demand_forecast should handle limit/sorting

            if not historical_data:
                print(f"  - No historical demand data found for Product {product_id}, Store {store_id}.")
                return 0, "No historical data." # Return 0 forecast if no history

            # Prepare better context for LLM
            data_summary, calculated_features = self._prepare_data_for_llm(historical_data)
            print(f"  - Prepared data summary/features for LLM:\n{data_summary[:500]}...") # Print start

            # Construct a more informed prompt
            prompt = f"""
            You are a retail demand forecasting expert analyzing Product ID '{product_id}' at Store ID '{store_id}'.
            Historical data context and key features:
            {data_summary}

            Considering the recent data, calculated features, especially '{calculated_features.get('latest_trend', 'N/A')}' trend and active promotions ('{calculated_features.get('latest_promo', 'N/A')}'),
            predict the total expected sales quantity for the next {forecast_horizon_days} days.

            Focus on the data patterns. Provide your prediction as a single integer number ONLY.
            Prediction:"""

            print(f"\n  - Sending refined prompt to Ollama (gemma:2b)...")

            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            raw_response_for_return = raw_response_content # Store for return
            print(f"  - Received raw response from Ollama: '{raw_response_content}'")

            # Parse the response (Using the robust parsing from before)
            predicted_demand = None
            print(f"  - DEBUG: Attempting to parse: '{raw_response_content}'")
            try:
                match1 = re.search(r'\b(?:is|prediction:?|Prediction:?)\s+(\d+)', raw_response_content, re.IGNORECASE)
                if match1:
                    predicted_demand = int(match1.group(1))
                    print(f"  - DEBUG: Matched Attempt 1 (is/prediction): {predicted_demand}")
                else:
                    # Attempt 2: Look for a number at the end
                    match2 = re.search(r'(\d+)\s*(?:units|items)?\.?\s*$', raw_response_content)
                    if match2:
                        predicted_demand = int(match2.group(1))
                        print(f"  - DEBUG: Matched Attempt 2 (end of string): {predicted_demand}")
                    else:
                        # Attempt 3: Last number fallback
                        all_numbers = re.findall(r'\d+', raw_response_content)
                        if all_numbers:
                            predicted_demand = int(all_numbers[-1])
                            print(f"  - DEBUG: Matched Attempt 3 (last number): {predicted_demand}")

                if predicted_demand is not None:
                    print(f"  - Parsed predicted demand: {predicted_demand}")
                    return predicted_demand, raw_response_for_return
                else:
                    # Fallback using calculated features if parsing fails
                    print(f"  - WARNING: No prediction number parsed from LLM. Using fallback.")
                    fallback_demand = round(calculated_features.get('avg_sales_last_7', calculated_features.get('avg_sales_last_N', 0)) * forecast_horizon_days)
                    print(f"  - Using fallback demand based on avg sales: {fallback_demand}")
                    return fallback_demand, raw_response_for_return

            except ValueError as ve:
                print(f"  - WARNING: Could not convert parsed value to integer. Error: {ve}")
                return None, raw_response_for_return
            except Exception as e:
                 print(f"  - ERROR during response parsing: {e}")
                 return None, raw_response_for_return

        except Exception as e:
            print(f"  - ERROR: Failed to get prediction from Ollama or process data: {e}")
            return None, f"Ollama Call/Processing Error: {e}"


# --- PricingAgent Class ---
class PricingAgent(BaseAgentLogic):
    def __init__(self, base_margin_percent=15.0):
        super().__init__(agent_name="PricingAgent")
        self.base_margin_percent = base_margin_percent
        self.role = 'Retail Pricing Strategist'
        self.goal = 'Recommend optimal pricing/discounts by analyzing inventory, demand, costs, competition, and other factors.'
        self.backstory = "Data-driven AI optimizing profitability and sales velocity via smart pricing, using rules and LLM reasoning."


    def _get_required_data(self, product_id: str, store_id: str):
        """ Fetches and combines data needed for pricing decisions. """
        print(f"  PricingAgent: Fetching data for Product {product_id}, Store {store_id}...")
        if not self.db_conn:
            print("  - ERROR: No database connection in _get_required_data.")
            return None

        try:
            product_id_int = int(product_id)
            store_id_int = int(store_id)

            # Use generic query for flexibility
            pricing_info_list = db_utils.get_generic_query(self.db_conn, "pricing_data", {"product_id": product_id_int, "store_id": store_id_int})
            pricing_info = pricing_info_list[0] if pricing_info_list else {}

            inventory_info_list = db_utils.get_inventory_levels(self.db_conn, product_id_int, store_id_int)
            inventory_info = inventory_info_list[0] if inventory_info_list else {}

            # Get *all* recent forecasts to potentially see trends or use latest valid one
            demand_info_list = db_utils.get_demand_forecast(self.db_conn, product_id=product_id_int, store_id=store_id_int) # gets latest first
            demand_info = demand_info_list[0] if demand_info_list else {} # Use latest for simplicity now


            # --- Cost Calculation ---
            # Refined Placeholder: Prioritize explicit cost if available, else estimate
            # TODO: Ideally, cost data should be in the database or passed in.
            cost_of_goods = None
            if pricing_info.get('cost_of_goods'): # Check if cost column exists/added
                 cost_of_goods = float(pricing_info['cost_of_goods'])
            elif pricing_info.get('price'):
                 cost_of_goods = float(pricing_info['price']) * 0.65 # Estimate: Cost = 65% of Price (adjust ratio)
            else:
                 cost_of_goods = 10.0 # Absolute fallback default cost

            storage_cost_val = pricing_info.get('storage_cost')
            if storage_cost_val is not None:
                 try: cost_of_goods += float(storage_cost_val)
                 except (ValueError, TypeError): print(f"  - WARNING: Could not add storage_cost '{storage_cost_val}'.")
            # --- /Cost Calculation ---

            combined_data = {
                "product_id": product_id_int,
                "store_id": store_id_int,
                "cost_of_goods": cost_of_goods, # Use calculated cost
                "current_price": float(pricing_info['price']) if pricing_info.get('price') is not None else None,
                "competitor_price": float(pricing_info['competitor_prices']) if pricing_info.get('competitor_prices') is not None else None,
                "current_discount": float(pricing_info.get('discounts', 0.0)),
                "sales_volume": int(pricing_info['sales_volume']) if pricing_info.get('sales_volume') is not None else None,
                "customer_reviews": pricing_info.get('customer_reviews'), # Could be rating (int/float) or text
                "return_rate": float(pricing_info['return_rate_']) if pricing_info.get('return_rate_') is not None else None,
                "storage_cost": float(pricing_info['storage_cost']) if pricing_info.get('storage_cost') is not None else None,
                "elasticity_index": float(pricing_info['elasticity_index']) if pricing_info.get('elasticity_index') is not None else None, # Higher = more price sensitive
                "current_stock": int(inventory_info['stock_levels']) if inventory_info.get('stock_levels') is not None else None,
                "reorder_point": int(inventory_info['reorder_point']) if inventory_info.get('reorder_point') is not None else None,
                "warehouse_capacity": int(inventory_info['warehouse_capacity']) if inventory_info.get('warehouse_capacity') is not None else None,
                "demand_trend": demand_info.get('demand_trend'), # From latest demand record
                "last_sales_qty" : int(demand_info['sales_quantity']) if demand_info.get('sales_quantity') is not None else None,
                "promotions_active": demand_info.get('promotions', 'No').lower() == 'yes',
                "seasonality": demand_info.get('seasonality_factors')
            }
            print(f"  PricingAgent: Combined data fetched (sample): cost={combined_data['cost_of_goods']:.2f}, price={combined_data['current_price']}, stock={combined_data['current_stock']}, comp_price={combined_data['competitor_price']}")
            return combined_data

        except (ValueError, TypeError) as e:
            print(f"  - ERROR: Invalid ID format or data conversion error in _get_required_data: {e}")
            return None
        except Exception as e:
            print(f"  - ERROR: An unexpected error occurred in _get_required_data: {e}")
            return None

    def _apply_pricing_rules(self, data):
        """ Applies more sophisticated rules based on combined data. """
        print("  PricingAgent: Applying pricing rules...")
        # Default to current state
        reason = "Maintain current price (default)."
        suggested_price = data.get('current_price')
        suggested_discount = data.get('current_discount', 0.0)

        if suggested_price is None: return {"new_price": None, "discount_percent": 0.0, "reason": "Missing current price."}

        min_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
        print(f"    - Rule Check: Min Price (Cost={data.get('cost_of_goods', 0.0):.2f} + {self.base_margin_percent}% margin) = {min_price:.2f}")

        # Rule Priorities (Example: Promotions > Stock Clearance > Competitor Matching > Demand Adjustments)

        # 1. Active Promotion Rule
        if data.get('promotions_active', False):
            promo_discount = 15.0 # Example standard promo discount - could be data driven
            potential_price = round(suggested_price * (1 - promo_discount / 100.0), 2)
            if potential_price >= min_price:
                suggested_discount = promo_discount
                reason = f"Applying standard {promo_discount}% promotion."
                print(f"    - Rule Applied: Promotion Active. New Discount: {suggested_discount}%")
            else:
                print(f"    - Rule Skipped: Promotion ({promo_discount}%) would violate minimum margin ({min_price:.2f}). Applying min price.")
                suggested_price = min_price # Apply min price directly
                suggested_discount = round((1 - min_price / data['current_price']) * 100.0, 1) if data['current_price'] > 0 else 0.0
                reason = f"Promotion Active, but discount limited by minimum margin. Set price to {min_price:.2f} ({suggested_discount}% effective discount)."
            # If promo applied, maybe skip other discount rules? Or let them stack? Let's skip others for now.
            final_price = round(suggested_price * (1 - suggested_discount / 100.0), 2) # Recalc needed if price was floor'd
            return {"new_price": max(final_price, min_price), "discount_percent": suggested_discount, "reason": reason}


        # 2. Inventory Level Rule (High Stock)
        stock = data.get('current_stock')
        reorder = data.get('reorder_point')
        trend = data.get('demand_trend')
        if stock is not None and reorder is not None and stock > reorder * 1.5: # High stock threshold
            if trend in ['Decreasing', 'Stable']:
                discount_increase = 10.0 # More aggressive discount for high stock
                potential_discount = suggested_discount + discount_increase # Assume stacking for now
                potential_price = round(data['current_price'] * (1 - potential_discount / 100.0), 2)

                if potential_price >= min_price:
                    suggested_discount = potential_discount
                    reason = f"High stock ({stock}) with {trend} demand. Increased discount to {suggested_discount}%."
                    print(f"    - Rule Applied: High Stock Discount. New Discount: {suggested_discount}%")
                else:
                    print(f"    - Rule Skipped: High stock discount ({potential_discount}%) would violate min margin.")
                    # Could choose to apply min price instead if desired


        # 3. Competitor Pricing Rule (Only if no promo/major discount already applied)
        comp_price = data.get('competitor_price')
        if comp_price is not None and suggested_discount < 5.0: # Only adjust if not already significantly discounted
            if suggested_price > comp_price * 1.01 : # If our price is notably higher
                potential_price = comp_price * 0.99 # Undercut slightly
                if potential_price >= min_price:
                    suggested_price = round(potential_price, 2)
                    suggested_discount = 0.0 # Reset discount
                    reason = f"Adjusting price to slightly undercut competitor ({comp_price:.2f})."
                    print(f"    - Rule Applied: Competitor Undercut. New Price: {suggested_price}")
                # Else: Can't undercut due to margin
            elif suggested_price < comp_price * 0.9: # If we are much cheaper
                # Maybe increase price slightly? Use elasticity? Needs careful logic.
                # Example: Increase towards competitor if elasticity allows
                elasticity = data.get('elasticity_index')
                if elasticity is not None and elasticity < 1.5: # If less elastic (less price sensitive)
                     potential_price = suggested_price * 1.03 # Increase by 3%
                     if potential_price < comp_price and potential_price >= min_price: # Ensure still below comp & above margin
                          suggested_price = round(potential_price, 2)
                          reason = f"Price significantly below competitor ({comp_price:.2f}) and low elasticity ({elasticity}). Increasing price slightly."
                          print(f"    - Rule Applied: Price Increase Opportunity. New Price: {suggested_price}")


        # 4. Demand Trend Rule (If no other major changes made)
        if reason == "Maintain current price (default).": # Only apply if nothing else changed price significantly
             if trend == 'Increasing':
                  # Maybe remove small discount or slightly increase price if possible?
                  if suggested_discount > 0:
                       suggested_discount = max(0.0, suggested_discount - 2.5) # Reduce discount slightly
                       reason = f"Demand Increasing. Reducing discount to {suggested_discount}%."
                       print(f"    - Rule Applied: Increasing Demand Discount Reduction. New Discount: {suggested_discount}%")
             elif trend == 'Decreasing':
                   # Add small discount if none exists?
                   if suggested_discount == 0.0:
                        potential_discount = 5.0
                        potential_price = round(data['current_price'] * (1 - potential_discount / 100.0), 2)
                        if potential_price >= min_price:
                             suggested_discount = potential_discount
                             reason = f"Demand Decreasing. Adding small discount ({suggested_discount}%)."
                             print(f"    - Rule Applied: Decreasing Demand Discount Add. New Discount: {suggested_discount}%")


        # Final Recalculation & Margin Check (repeated for safety)
        final_price = round(data['current_price'] * (1 - suggested_discount / 100.0), 2)
        if final_price < min_price:
             final_price = min_price
             if data['current_price'] > 0: suggested_discount = round((1 - min_price / data['current_price']) * 100.0, 1)
             else: suggested_discount = 0.0
             reason += f" Final adjustment to meet minimum margin floor ({min_price:.2f})."
             print(f"    - Final price adjusted to minimum margin: {final_price}. Effective discount: {suggested_discount}%")

        print(f"  PricingAgent: Rules finished. Final Reason: {reason}")
        return {"new_price": final_price, "discount_percent": suggested_discount, "reason": reason}


    def _get_ollama_price_recommendation(self, data, rule_recommendation):
        """ Sends pricing context AND rule suggestion to Ollama and parses recommendation. """
        print("  PricingAgent: Preparing data and prompt for Ollama validation/refinement...")

        # Format rule recommendation for prompt
        rule_reason = rule_recommendation.get('reason', 'N/A')
        rule_price = rule_recommendation.get('new_price')
        rule_discount = rule_recommendation.get('discount_percent')
        # Ensure values are formatted reasonably if None
        rule_price_str = f"{rule_price:.2f}" if rule_price is not None else "N/A"
        rule_discount_str = f"{rule_discount:.1f}%" if rule_discount is not None else "N/A"
        rule_summary = f"Our rule-based system suggested: Target Price={rule_price_str}, Effective Discount={rule_discount_str}, Reason=[{rule_reason}]."

        # Include more context from data dictionary
        prompt = f"""
        You are an expert Retail Pricing Strategist AI providing analysis.
        Analyze the following data for Product ID '{data['product_id']}' at Store ID '{data['store_id']}':
        - Estimated Cost of Goods: {data.get('cost_of_goods', 'N/A'):.2f}
        - Current Price: {data.get('current_price', 'N/A')}
        - Current Stock Level: {data.get('current_stock', 'N/A')} (Reorder Point: {data.get('reorder_point', 'N/A')})
        - Demand Trend (Recent): {data.get('demand_trend', 'N/A')}
        - Last Sales Qty (Snapshot): {data.get('last_sales_qty', 'N/A')}
        - Competitor Price: {data.get('competitor_price', 'N/A')}
        - Elasticity Index: {data.get('elasticity_index', 'N/A')} (Higher value means more price sensitive)
        - Storage Cost: {data.get('storage_cost', 'N/A')}
        - Return Rate %: {data.get('return_rate', 'N/A')}
        - Promotion Currently Active: {data.get('promotions_active', False)}
        - Seasonality Factor: {data.get('seasonality', 'N/A')}
        - Customer Reviews Info: {data.get('customer_reviews', 'N/A')}

        Rule-Based Context: {rule_summary}

        Based on *all* the provided data (especially considering elasticity, competitor price, inventory, demand trend, and potential impact of reviews/returns):
        1. Briefly state if you agree with the rule-based suggestion (True/False).
        2. Provide your final optimized recommendation as a JSON object like this (ONLY the JSON object, no other text):
        {{
          "llm_reason": "Your brief reasoning focusing on factors rules might miss or confirming the rules.",
          "llm_agrees_with_rule": <true_or_false>,
          "llm_suggested_price": <price_number_or_null_if_discount_focused>,
          "llm_suggested_discount_percent": <discount_percentage_or_null_if_price_focused>
        }}
        """
        print("  - Sending refined prompt to Ollama (gemma:2b)...")

        try:
            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            print(f"  - Received raw response from Ollama: {raw_response_content}")

            # Parse the JSON response robustly
            try:
                # Find JSON block using regex, even if surrounded by text/markdown
                json_match = re.search(r'\{.*?\}', raw_response_content, re.DOTALL | re.MULTILINE)
                if json_match:
                    json_string = json_match.group(0)
                    # Attempt to repair slightly malformed JSON if needed (optional)
                    # try: from json_repair import repair_json except ImportError: repair_json = lambda x: x
                    # recommendation = json.loads(repair_json(json_string))
                    recommendation = json.loads(json_string)

                    # Basic validation
                    if 'llm_reason' in recommendation and 'llm_agrees_with_rule' in recommendation:
                         print(f"  - Parsed Ollama recommendation: {recommendation}")
                         # Ensure numeric fields are numbers or None
                         recommendation['llm_suggested_price'] = float(recommendation['llm_suggested_price']) if recommendation.get('llm_suggested_price') is not None else None
                         recommendation['llm_suggested_discount_percent'] = float(recommendation['llm_suggested_discount_percent']) if recommendation.get('llm_suggested_discount_percent') is not None else None
                         return recommendation
                    else:
                         print("  - WARNING: Parsed JSON lacks required keys ('llm_reason', 'llm_agrees_with_rule').")
                         return {"llm_reason": "Parsed JSON lacks required keys.", "llm_agrees_with_rule": None}
                else:
                     print("  - WARNING: No JSON object found in Ollama response.")
                     return {"llm_reason": "No JSON object found in response.", "llm_agrees_with_rule": None}

            except json.JSONDecodeError as jde:
                print(f"  - WARNING: Could not parse JSON response from Ollama: {jde}")
                return {"llm_reason": f"JSON parsing error: {jde}", "llm_agrees_with_rule": None}
            except Exception as e:
                 print(f"  - ERROR during JSON parsing or processing: {e}")
                 return {"llm_reason": f"Parsing exception: {e}", "llm_agrees_with_rule": None}

        except Exception as e:
            print(f"  - ERROR: Failed to get recommendation from Ollama: {e}")
            return {"llm_reason": f"Ollama call failed: {e}", "llm_agrees_with_rule": None}


def recommend_price(self, product_id: str, store_id: str) -> dict:
    """Recommends price/discount using a hybrid approach with web-scraped market data, returns dict."""
    print(f"\nPricingAgent: Generating HYBRID price recommendation for Product {product_id}, Store {store_id}...")
    final_recommendation = {
        "recommendation": "Error: Processing failed.",
        "final_price": None, "final_discount": None,
        "rule_reason": None, "llm_reason": None, "market_reason": None
    }

    # 1. Fetch internal data
    data = self._get_required_data(product_id, store_id)
    if not data or data.get('current_price') is None:
        print("  - ERROR: Insufficient data for pricing analysis.")
        final_recommendation["recommendation"] = "Error: Insufficient data"
        return final_recommendation

    # 2. Fetch web-scraped market data
    try:
        conn = sqlite3.connect('ecommerce_data.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(price) FROM products WHERE product_name LIKE ?", (f"%{product_id}%",))
        avg_market_price = cursor.fetchone()[0]  # Returns None if no data
        conn.close()
        
        if avg_market_price:
            print(f"  - Web-Scraped Market Data: Average price from e-commerce sites = ₹{avg_market_price:.2f}")
            final_recommendation["market_reason"] = f"Average market price from web scraping: ₹{avg_market_price:.2f}"
        else:
            print("  - No web-scraped market data available for product {product_id}.")
            final_recommendation["market_reason"] = "No market data available from web scraping."
    except Exception as e:
        print(f"  - ERROR fetching web-scraped data: {e}")
        final_recommendation["market_reason"] = f"Error accessing market data: {e}"

    # 3. Apply Rules
    rule_recommendation = self._apply_pricing_rules(data)
    rule_price = rule_recommendation.get('new_price')
    rule_discount = rule_recommendation.get('discount_percent')
    final_recommendation["rule_reason"] = rule_recommendation.get('reason')

    # 4. Get LLM Validation/Refinement
    ollama_opinion = self._get_ollama_price_recommendation(data, rule_recommendation)
    final_recommendation["llm_reason"] = ollama_opinion.get('llm_reason')
    llm_price = ollama_opinion.get('llm_suggested_price')
    llm_discount = ollama_opinion.get('llm_suggested_discount_percent')

    # 5. Reconcile Rule, LLM, and Market Data
    current_price = data.get('current_price')
    min_price = self.min_price  # Assuming this is set in PricingAgent (e.g., cost + margin)

    # Default to rule-based pricing
    final_price = rule_price
    final_discount = rule_discount

    # Adjust based on market data if available
    if avg_market_price:
        # Aim to be competitive: 5% below market average, but respect min_price
        market_adjusted_price = max(min_price, avg_market_price * 0.95)
        if market_adjusted_price < current_price:
            final_price = market_adjusted_price
            final_discount = min(15.0, (current_price - final_price) / current_price * 100) if current_price else 10.0
            final_recommendation["market_reason"] += f"; Adjusted to ₹{final_price:.2f} (5% below market avg)."
        else:
            final_recommendation["market_reason"] += "; Market price too high, sticking with rule-based."

    # Optional: Override with LLM if it disagrees and respects min_price
    if ollama_opinion.get('llm_agrees_with_rule') is False and llm_price and llm_price >= min_price:
        final_price = llm_price
        final_discount = llm_discount if llm_discount is not None else final_discount
        final_recommendation["llm_reason"] += "; LLM override applied."

    # Set final values
    final_recommendation["final_price"] = final_price
    final_recommendation["final_discount"] = final_discount

    # 6. Construct recommendation string
    combined_reason = (
        f"Rule Action: [{final_recommendation['rule_reason'] or 'N/A'}]. "
        f"LLM Analysis: [{final_recommendation['llm_reason'] or 'N/A'}]. "
        f"Market Insight: [{final_recommendation['market_reason'] or 'N/A'}]"
    )
    final_recommendation["recommendation"] = combined_reason if final_recommendation["rule_reason"] or final_recommendation["llm_reason"] or final_recommendation["market_reason"] else "No specific recommendation generated."

    print(f"  - Pricing Recommendation: {final_recommendation['recommendation']}")
    print(f"  - Final Suggested Price: {final_recommendation['final_price']}, Final Suggested Discount: {final_recommendation['final_discount']}%")

    return final_recommendation# Return the full dictionary

    # close_connection is inherited from BaseAgentLogic