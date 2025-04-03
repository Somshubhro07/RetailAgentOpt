# dynamic_price_fetcher.py
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import re

# Ethical scraping settings
DELAY_SECONDS = 5
HEADERS = {'User-Agent': UserAgent().random}

# Check robots.txt (simplified for dynamic use)
def check_robots_txt(url, target_path):
    try:
        robots_url = f"{url}/robots.txt"
        response = requests.get(robots_url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"No robots.txt found for {url}, proceeding cautiously.")
            return True
        
        lines = response.text.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("User-agent: *") and "Disallow: /search" in line:
                print(f"Search scraping disallowed by {url}'s robots.txt.")
                return False
        return True
    except Exception as e:
        print(f"Error checking robots.txt for {url}: {e}")
        return True

# Fetch price dynamically based on user query
def fetch_price_for_query(query: str, site="flipkart"):
    print(f"Fetching price for query '{query}' from {site}...")
    
    if site == "flipkart":
        base_url = "https://www.flipkart.com"
        search_url = f"{base_url}/search?q={query.replace(' ', '+')}"
    else:
        return None  # Add more sites as needed

    # Check robots.txt
    if not check_robots_txt(base_url, "/search"):
        print(f"Skipping {site} due to robots.txt restrictions.")
        return None

    # Selenium setup
    options = Options()
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(search_url)
        time.sleep(5)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        products = soup.select('div[class*="_1AtVbE"]')[:3]  # First 3 results
        name_selector = 'div[class*="_4rR01T"]'
        price_selector = 'div[class*="_30jeq3"]'
        
        prices = []
        for product in products:
            try:
                name = product.select_one(name_selector)
                price = product.select_one(price_selector)
                product_name = name.text.strip() if name else "Unknown"
                price_text = price.text.strip() if price else "0"
                price_text = re.sub(r'[^0-9.]', '', price_text)
                price_value = float(price_text or 0)
                print(f"Found: {product_name} - ₹{price_value}")
                prices.append(price_value)
            except Exception as e:
                print(f"Error parsing product: {e}")
        
        avg_price = sum(prices) / len(prices) if prices else None
        return avg_price
    
    except Exception as e:
        print(f"Error fetching price from {site}: {e}")
        return None
    finally:
        driver.quit()
        time.sleep(DELAY_SECONDS)

if __name__ == "__main__":
    avg_price = fetch_price_for_query("bluetooth speaker")
    print(f"Average price for 'bluetooth speaker': ₹{avg_price:.2f}" if avg_price else "No price data found.")