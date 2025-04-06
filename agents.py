# agents.py
import re
from crewai import Agent
import ollama
import db_utils
import pandas as pd
import json

# --- InventoryAgent Class ---
class InventoryAgent:
    def __init__(self):
        self.agent = Agent(
            role='Inventory Management Specialist',
            goal='Monitor stock levels, check reorder points, identify expiring stock, and request restocking based on demand forecasts and current inventory.',
            backstory=(
                "An meticulous AI agent responsible for tracking every item across all stores and warehouses. "
                "It ensures shelves are optimally stocked by analyzing current levels against reorder points, "
                "considering supplier lead times and product expiry dates. It communicates stock status and needs "
                "proactively to prevent stockouts and minimize waste."
            ),
            verbose=True,
            allow_delegation=False
        )
        print("InventoryAgent initialized.")

    def check_stock_status(self, product_id=None, store_id=None, conn=None):
        """ Checks current stock levels against reorder points and expiry dates. """
        print(f"\nInventoryAgent: Checking stock status for product: {product_id or 'All'}, store: {store_id or 'All'}...")
        if not conn:
            print("Error: No database connection available.")
            return {"low_stock": [], "expiring_soon": []}

        try:
            inventory_data = db_utils.get_inventory_levels(conn, product_id, store_id)
            if not inventory_data:
                print("No inventory data found for the specified criteria.")
                return {"low_stock": [], "expiring_soon": []}

            low_stock_items = []
            expiring_items = []

            for item in inventory_data:
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

            print(f"Stock check complete. Low stock items: {len(low_stock_items)}, Expiring items: {len(expiring_items)}")
            return {
                "low_stock": low_stock_items,
                "expiring_soon": expiring_items
            }
        except Exception as e:
            print(f"An error occurred during stock status check: {e}")
            return {"low_stock": [], "expiring_soon": []}

    def update_inventory_after_sales(self, sales_data, conn=None):
        """ Updates inventory levels based on sales data. """
        print("\nInventoryAgent: Updating inventory based on sales data...")
        if not conn:
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

            current_inventory = db_utils.get_inventory_levels(conn, product_id=pid, store_id=sid)
            if not current_inventory:
                print(f"  - WARNING: No inventory record found for Product {pid}, Store {sid}. Cannot update stock.")
                continue

            inv_record = current_inventory[0]
            current_stock = inv_record.get('stock_levels')
            if current_stock is None:
                print(f"  - WARNING: Stock level is null for Product {pid}, Store {sid}. Cannot update.")
                continue

            new_stock = current_stock - qty_sold
            if new_stock < 0:
                print(f"  - WARNING: Sale of {qty_sold} for Product {pid}, Store {sid} results in negative stock ({new_stock}). Setting stock to 0.")
                new_stock = 0

            if db_utils.update_stock_level(conn, pid, sid, new_stock):
                success_count += 1
            else:
                print(f"  - Failed to update stock for Product {pid}, Store {sid}.")

        print(f"Inventory update complete. {success_count}/{len(sales_data)} sales records processed.")
        return success_count == len(sales_data)

    def calculate_order_request(self, product_id, store_id, demand_forecast, conn=None):
        """ Calculates order request based on stock and demand forecast. """
        print(f"\nInventoryAgent: Calculating order request for Product {product_id}, Store {store_id} with forecast {demand_forecast}...")
        order_request = {"product_id": product_id, "store_id": store_id, "order_quantity": 0, "reason": "No order needed."}
        if not conn:
            print("Error: No database connection.")
            order_request["reason"] = "Error: DB connection failed."
            return order_request

        try:
            product_id_int = int(product_id)
            store_id_int = int(store_id)
            inventory_data = db_utils.get_inventory_levels(conn, product_id_int, store_id_int)
            if not inventory_data:
                order_request["reason"] = "Error: No inventory data found."
                return order_request

            item = inventory_data[0]
            stock = item.get('stock_levels', 0)
            reorder_point = item.get('reorder_point', 0)
            lead_time = item.get('supplier_lead_time_days', 3)
            capacity = item.get('warehouse_capacity', 1000)

            daily_demand_estimate = demand_forecast / 7 if demand_forecast > 0 else 0
            demand_during_lead_time = daily_demand_estimate * lead_time
            safety_stock = reorder_point
            target_stock_level = demand_during_lead_time + safety_stock

            needed_quantity = max(0, target_stock_level - stock)
            suggested_order = 0

            if needed_quantity > 0:
                suggested_order = max(50, round(needed_quantity))
                reason = f"Stock ({stock}) below target ({target_stock_level:.0f}). Need ~{needed_quantity:.0f} units."

                available_capacity = capacity - stock
                if suggested_order > available_capacity:
                    suggested_order = available_capacity
                    reason += f" Order clamped to available capacity ({available_capacity})."

                order_request["order_quantity"] = suggested_order
                order_request["reason"] = reason
                print(f"  - Order Request: {suggested_order} units for {product_id}/{store_id}. Reason: {reason}")
            else:
                order_request["reason"] = f"Stock ({stock}) is at or above target level ({target_stock_level:.0f})."

            return order_request
        except Exception as e:
            print(f"An error occurred during order request calculation: {e}")
            order_request["reason"] = f"Error: {e}"
            return order_request

# --- DemandForecastAgent Class ---
class DemandForecastAgent:
    def __init__(self):
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
            allow_delegation=False
        )
        print("DemandForecastAgent initialized.")

    def _prepare_data_for_llm(self, history_data, recent_records=30):
        if not history_data:
            return "No historical data available."

        df = pd.DataFrame(history_data)
        recent_df = df.head(recent_records)
        summary_data = recent_df[[
            'date', 'sales_quantity', 'price', 'promotions', 'seasonality_factors', 'demand_trend'
        ]].to_dict(orient='records')
        return json.dumps(summary_data, indent=2)

    def generate_forecast(self, product_id, store_id, forecast_horizon_days=7, conn=None):
        print(f"\nDemandForecastAgent: Generating forecast for Product {product_id}, Store {store_id} for next {forecast_horizon_days} days...")
        if not conn:
            print("Error: No database connection.")
            return None, "Error: No DB connection."

        historical_data = db_utils.get_demand_forecast(conn, product_id=product_id, store_id=store_id)
        if not historical_data:
            print(f"  - No historical demand data found for Product {product_id}, Store {store_id}.")
            return 0, "No historical data."

        data_summary = self._prepare_data_for_llm(historical_data)
        print(f"  - Prepared data summary for LLM:\n{data_summary[:500]}...")

        prompt = f"""
        You are a retail demand forecasting expert.
        Analyze the following recent historical sales data for Product ID '{product_id}' at Store ID '{store_id}':
        {data_summary}

        Based *only* on the trends, seasonality, price points, and promotions visible in this data,
        predict the total expected sales quantity for the next {forecast_horizon_days} days.

        Provide your prediction as a single integer number only, without any explanation or extra text.
        Prediction:
        """
        print(f"  - Sending prompt to Ollama (gemma:2b)...")

        try:
            response = ollama.chat(
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            print(f"  - Received raw response from Ollama: '{raw_response_content}'")

            predicted_demand = None
            match1 = re.search(r'\b(?:is|prediction:?)\s+(\d+)', raw_response_content, re.IGNORECASE)
            if match1:
                predicted_demand = int(match1.group(1))
            else:
                match2 = re.search(r'(\d+)\s*(?:units|items)?\.?\s*$', raw_response_content)
                if match2:
                    predicted_demand = int(match2.group(1))
                else:
                    all_numbers = re.findall(r'\d+', raw_response_content)
                    predicted_demand = int(all_numbers[-1]) if all_numbers else None

            if predicted_demand is not None:
                print(f"  - Parsed predicted demand: {predicted_demand}")
                return predicted_demand, raw_response_content
            else:
                print(f"  - WARNING: No prediction number parsed. Using fallback.")
                fallback_demand = 0
                if len(historical_data) >= 7:
                    fallback_demand = int(pd.DataFrame(historical_data).head(7)['sales_quantity'].mean() * forecast_horizon_days)
                elif historical_data:
                    fallback_demand = int(pd.DataFrame(historical_data)['sales_quantity'].mean() * forecast_horizon_days)
                print(f"  - Using fallback demand: {fallback_demand}")
                return fallback_demand, raw_response_content
        except Exception as e:
            print(f"  - ERROR: Failed to get prediction from Ollama: {e}")
            return None, f"Ollama Call Error: {e}"

# --- PricingAgent Class ---
class PricingAgent:
    def __init__(self, base_margin_percent=15.0):
        self.base_margin_percent = base_margin_percent
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
        )
        print("PricingAgent initialized.")

    def _get_required_data(self, product_id, store_id, conn=None):
        print(f"  PricingAgent: Fetching data for Product {product_id}, Store {store_id}...")
        if not conn:
            print("  - ERROR: No database connection in _get_required_data.")
            return None

        try:
            product_id_int = int(product_id)
            store_id_int = int(store_id)

            pricing_info_list = db_utils.get_generic_query(conn, "pricing_data", {"product_id": product_id_int, "store_id": store_id_int})
            pricing_info = pricing_info_list[0] if pricing_info_list else {}

            inventory_info_list = db_utils.get_inventory_levels(conn, product_id_int, store_id_int)
            inventory_info = inventory_info_list[0] if inventory_info_list else {}

            demand_info_list = db_utils.get_demand_forecast(conn, product_id_int, store_id_int)
            demand_info = demand_info_list[0] if demand_info_list else {}

            combined_data = {
                "product_id": product_id_int,
                "store_id": store_id_int,
                "current_price": pricing_info.get('price'),
                "competitor_price": pricing_info.get('competitor_prices'),
                "current_discount": pricing_info.get('discounts', 0.0),
                "sales_volume": pricing_info.get('sales_volume'),
                "customer_reviews": pricing_info.get('customer_reviews'),
                "return_rate": pricing_info.get('return_rate_'),
                "storage_cost": pricing_info.get('storage_cost'),
                "elasticity_index": pricing_info.get('elasticity_index'),
                "current_stock": inventory_info.get('stock_levels'),
                "reorder_point": inventory_info.get('reorder_point'),
                "demand_trend": demand_info.get('demand_trend'),
                "last_sales_qty": demand_info.get('sales_quantity'),
                "promotions_active": demand_info.get('promotions', 'No') == 'Yes',
                "seasonality": demand_info.get('seasonality_factors')
            }

            cost_of_goods = combined_data.get('current_price', 0.0) * 0.7 if combined_data.get('current_price') is not None else 10.0
            storage_cost_val = combined_data.get('storage_cost')
            if storage_cost_val is not None:
                try:
                    cost_of_goods += float(storage_cost_val)
                except (ValueError, TypeError):
                    print(f"  - WARNING: Could not add storage_cost '{storage_cost_val}' to cost_of_goods.")
            combined_data['cost_of_goods'] = cost_of_goods

            print(f"  PricingAgent: Combined data fetched (sample): cost_of_goods={combined_data['cost_of_goods']:.2f}, current_price={combined_data['current_price']}, current_stock={combined_data['current_stock']}")
            return combined_data
        except (ValueError, TypeError) as e:
            print(f"  - ERROR: Invalid ID format provided? Could not convert '{product_id}' or '{store_id}' to integer. Error: {e}")
            return None
        except Exception as e:
            print(f"  - ERROR: An unexpected error occurred in _get_required_data: {e}")
            return None

    def _apply_pricing_rules(self, data):
        print("  PricingAgent: Applying pricing rules...")
        reason = "Maintain current price (default)."
        suggested_price = data.get('current_price')
        suggested_discount = data.get('current_discount', 0.0)

        if suggested_price is None:
            return {"new_price": None, "discount_percent": 0.0, "reason": "Missing current price."}

        min_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
        print(f"    - Rule Check: Min Price (Cost + {self.base_margin_percent}% margin) = {min_price:.2f}")

        if data.get('competitor_price') and suggested_price > data['competitor_price']:
            potential_price = data['competitor_price'] * 0.99
            if potential_price >= min_price:
                suggested_price = round(potential_price, 2)
                suggested_discount = 0.0
                reason = f"Price adjusted to slightly undercut competitor ({data['competitor_price']:.2f})."
                print(f"    - Rule Applied: Competitor Undercut. New Price: {suggested_price}")
            else:
                print(f"    - Rule Skipped: Undercutting competitor would violate minimum margin.")

        stock = data.get('current_stock')
        reorder = data.get('reorder_point')
        trend = data.get('demand_trend')
        if stock and reorder and stock > reorder * 1.5:
            if trend in ['Decreasing', 'Stable']:
                discount_increase = 5.0
                potential_discount = suggested_discount + discount_increase
                potential_price = round(data['current_price'] * (1 - potential_discount / 100.0), 2)

                if potential_price >= min_price:
                    suggested_discount = potential_discount
                    reason = f"High stock ({stock}) with {trend} demand. Increased discount to {suggested_discount}%."
                    print(f"    - Rule Applied: High Stock Discount. New Discount: {suggested_discount}%")
                else:
                    print(f"    - Rule Skipped: High stock discount would violate minimum margin.")

        if data.get('promotions_active', False):
            promo_discount = 15.0
            potential_price = round(data['current_price'] * (1 - promo_discount / 100.0), 2)
            if potential_price >= min_price:
                suggested_discount = promo_discount
                reason = f"Applying standard {promo_discount}% promotion."
                print(f"    - Rule Applied: Promotion Active. New Discount: {suggested_discount}%")
            else:
                print(f"    - Rule Skipped: Promotion discount would violate minimum margin.")

        final_price = round(data['current_price'] * (1 - suggested_discount / 100.0), 2)
        if final_price < min_price:
            final_price = min_price
            if data['current_price'] > 0:
                suggested_discount = round((1 - final_price / data['current_price']) * 100.0, 1)
            reason += f" Adjusted to meet minimum margin floor."
            print(f"    - Final price adjusted to minimum margin: {final_price}. Effective discount: {suggested_discount}%")

        print(f"  PricingAgent: Rules finished. Reason: {reason}")
        return {"new_price": final_price, "discount_percent": suggested_discount, "reason": reason}

    def _get_ollama_price_recommendation(self, data, rule_recommendation, conn=None):
        print("  PricingAgent: Preparing data and prompt for Ollama validation/refinement...")
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
                model='gemma:2b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_response_content = response['message']['content'].strip()
            print(f"  - Received raw response from Ollama: {raw_response_content}")

            json_match = re.search(r'\{.*\}', raw_response_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                recommendation = json.loads(json_string)
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
            print(f"  - ERROR: Failed to get recommendation from Ollama: {e}")
            return {"llm_reason": f"Ollama call failed: {e}", "llm_agrees_with_rule": None}

    def recommend_price(self, product_id, store_id, conn=None):
        print(f"\nPricingAgent: Generating HYBRID price recommendation for Product {product_id}, Store {store_id}...")
        data = self._get_required_data(product_id, store_id, conn)
        if not data or data.get('current_price') is None:
            print("  - ERROR: Insufficient data.")
            return {"recommendation": "Error: Insufficient data", "final_price": None, "final_discount": None, "rule_reason": None, "llm_reason": None}

        rule_recommendation = self._apply_pricing_rules(data)
        ollama_opinion = self._get_ollama_price_recommendation(data, rule_recommendation, conn)

        final_price = rule_recommendation.get('new_price')
        final_discount = rule_recommendation.get('discount_percent')
        combined_reason = f"Rule Action: [{rule_recommendation.get('reason', 'N/A')}]. LLM Analysis: [{ollama_opinion.get('llm_reason', 'N/A')}]"

        # Simple override if LLM disagrees and provides a suggestion
        if ollama_opinion.get('llm_agrees_with_rule') is False:
            llm_price = ollama_opinion.get('llm_suggested_price')
            llm_discount = ollama_opinion.get('llm_suggested_discount_percent')
            if llm_price is not None and llm_price >= data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0):
                final_price = llm_price
                final_discount = 0.0
                combined_reason += " [Decision: Used LLM price override]"
            elif llm_discount is not None:
                final_discount = llm_discount
                final_price = round(data['current_price'] * (1 - final_discount / 100.0), 2)
                if final_price < data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0):
                    final_price = data.get('cost_of_goods', 0.0) * (1 + self.base_margin_percent / 100.0)
                    final_discount = round((1 - final_price / data['current_price']) * 100.0, 1)
                combined_reason += " [Decision: Used LLM discount override]"

        print(f"  - Pricing Recommendation: {combined_reason}")
        print(f"  - Final Suggested Price: {final_price}, Final Suggested Discount: {final_discount}%")

        return {
            "recommendation": combined_reason,
            "final_price": final_price,
            "final_discount": final_discount,
            "rule_reason": rule_recommendation.get('reason'),
            "llm_reason": ollama_opinion.get('llm_reason')
        }