# 🚀 Retail Inventory Optimizer 🛒💨

**(Built for the Gen AI Hackathon)**

Tired of inventory chaos? Drowning in overstock while simultaneously losing sales to stockouts? Yeah, we thought so.

Welcome to **Retail Inventory Optimizer**, your new best friend in the fight against inventory mismanagement. This isn't just another boring spreadsheet; it's a slick, multi-agent AI system powered by on-premise LLMs (**Ollama** with `gemma:2b`), **CrewAI**, and a nifty **Flask** web interface. We're here to predict demand, slash holding costs, nail pricing, and make your supply chain smoother than a fresh jar of Skippy.

---

## 😩 The Problem We're Crushing

Let's be real, retail inventory is a battlefield. You're constantly dodging bullets like:

* **👻 Stockouts:** Poof! Popular items vanish, taking potential sales with them. Frustrating, right?
* **📦 Overstocking:** Mountains of unsold goods silently draining your profits via holding costs. Ouch.
* **🤔 Inefficient Pricing:** Leaving money on the table with pricing that doesn't react to reality.
* **🐌 Supply Chain Snails:** Lack of smooth communication leading to costly delays.

---

## ✨ Our Genius Solution ✨

We're throwing AI agents at the problem! Our **Retail Inventory Optimizer** uses a crack team of specialized agents built with **CrewAI** that collaborate like pros:

* 🔮 **DemandForecastAgent**: Predicts future demand using historical data and real-world events (holidays, weather, you name it). It even gets refinement tips from our `gemma:2b` LLM.
* 📊 **InventoryAgent**: Keeps a hawk-eye on stock levels. If things get low, it taps the SupplierAgent on the shoulder.
* 💰 **PricingAgent**: The mastermind of markdowns and markups. Analyzes demand, stock, *and* customer whining (sentiment) to suggest prices that actually work. Uses `gemma:2b` for sentiment analysis and recommendation refinement.
* 🚚 **SupplierAgent**: Automates the boring task of reordering, ensuring suppliers are looped in *before* disaster strikes.

**Under the Hood:**
* 🧠 **Ollama LLM (`gemma:2b`)**: On-premise brainpower for smart forecasting, pricing, and sentiment checks.
* 💾 **SQLite Database**: The memory bank holding sales, inventory, feedback, and external event data.
* 🌐 **Flask Web App**: A sleek, simple UI to control the magic and see the results.

---

## 🚀 Killer Features

* **Crystal Ball Demand Forecasting**: See future sales before they happen.
* **Zero-Hassle Inventory Management**: Automated low-stock alerts and reordering.
* **Whip-Smart Dynamic Pricing**: Prices that adapt to demand, stock levels, and customer moods.
* **Smooth Supplier Collab**: Proactive reordering means fewer "Oops, we're out!" moments.
* **Know-Your-Customer Sentiment Analysis**: Turns feedback into actionable pricing strategies.
* **Dead-Simple Web Interface**: Interact with the system without needing a PhD in AI.

---

## 🛠️ Tech Stack Arsenal

We built this beast with some seriously cool tech:

* **Backend:**
    * Python 3.8+
    * Flask (The lightweight web framework champ)
    * CrewAI (For orchestrating our agent squad)
    * LangChain-Ollama (Connecting to our local LLM brain)
    * SQLite3 (Reliable data storage, no fuss)
* **AI Brain:**
    * Ollama (`gemma:2b` model) - Keeping it local and powerful!
* **Frontend:**
    * HTML / CSS / JavaScript (The classic trio)
    * Chart.js (Optional, for making data look pretty 📊)
* **Database:**
    * SQLite

---

## 📂 Project Structure

Here's how the project is organized:

```
RetailAgentopt/
├── agents.py             # Core logic for AI agents (Demand, Inventory, Pricing, Supplier)
├── sentiment_analyzer.py # Sentiment analysis module (powered by Ollama LLM)
├── app.py                # Flask backend: API endpoints and app logic
├── ecommerce_data.sqlite # SQLite database storing sales, inventory, and feedback data
├── templates/
│   ├── index.html        # Main HTML template for the web interface
│   └── style.css         # Styling for the frontend
├── static/
│   ├── js/
│   │   └── script.js     # JavaScript for frontend interactivity
│   └── css/
│       └── style.css     # Additional CSS for frontend design
├── requirements.txt      # Python dependencies for the project
└── README.md             # Project documentation (this file)
```

Each file and folder plays a crucial role in making the **Retail Inventory Optimizer** work seamlessly. Dive in and explore!


---

## 🖥️ Setup: Let's Get This Running!

Ready to fire it up? Let's go.

**Prerequisites:** You absolutely *need* these:
* Python 3.8+ (Don't be using ancient versions!)
* SQLite (Usually built-in, but check)
* Ollama installed and **running**. Seriously, make sure it's up.
* The `gemma:2b` Ollama model downloaded.
* A modern web browser (Chrome recommended if using Chart.js).

**Installation Steps:**

1.  **Clone This Bad Boy:**
    ```bash
    git clone [https://github.com/your-username/RetailAgentopt.git](https://github.com/your-username/RetailAgentopt.git)
    cd RetailAgentopt
    ```

2.  **Virtual Environment Goodness (Do It!):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get Ollama Ready:**
    * Confirm Ollama is running (usually at `http://localhost:11434`).
    * Pull the model if you haven't already (this might take a bit):
        ```bash
        ollama pull gemma:2b
        ```
    * *Troubleshooting:* If the app can't connect later, 99% chance Ollama isn't running or the model isn't pulled.

5.  **Initialize the Database (If it doesn't exist):**
    * The `ecommerce_data.sqlite` file should contain the necessary tables. If you need to create it from scratch or reset it:
    ```bash
    # Run these commands in your terminal
    sqlite3 ecommerce_data.sqlite <<EOF
    DROP TABLE IF EXISTS sales;
    DROP TABLE IF EXISTS inventory;
    DROP TABLE IF EXISTS customer_feedback;
    DROP TABLE IF EXISTS external_factors;

    CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id TEXT, store_id TEXT, sales INTEGER, date TEXT);
    CREATE TABLE inventory (id INTEGER PRIMARY KEY, product_id TEXT, store_id TEXT, stock_level INTEGER);
    CREATE TABLE customer_feedback (id INTEGER PRIMARY KEY, product_id TEXT, feedback TEXT, sentiment TEXT, timestamp TEXT);
    CREATE TABLE external_factors (id INTEGER PRIMARY KEY, date TEXT, factor_type TEXT, factor_value TEXT, impact REAL);

    -- Add some starter data
    INSERT INTO sales (product_id, store_id, sales, date) VALUES ('9286', '16', 50, '2025-04-01');
    INSERT INTO inventory (product_id, store_id, stock_level) VALUES ('9286', '16', 700);
    INSERT INTO customer_feedback (product_id, feedback, sentiment, timestamp) VALUES ('9286', 'This Bluetooth Speaker is too expensive for its quality.', 'negative', '2025-04-01');
    INSERT INTO external_factors (date, factor_type, factor_value, impact) VALUES ('2025-04-01', 'holiday', 'Diwali', 1.5);
    -- Add more sample data as needed!
    .quit
    EOF
    echo "Database initialized!"
    ```
    *(Note: The above SQL block creates tables and adds sample data. Adjust sample data as needed.)*

6.  **Run the Flask App:**
    ```bash
    python app.py
    ```

7.  **Access the App:**
    Open your browser and navigate to `http://127.0.0.1:5000/`. Boom!

---

## 🌐 How to Use It

It's designed to be easy:

1.  Go to `http://127.0.0.1:5000/`.
2.  Punch in a `product_id` (like `9286`), `store_id` (like `16`), and how many `days` you want to forecast (like `7`).
3.  Hit the button!
4.  Check out the results: demand forecast, inventory status (and if an order was placed), plus the oh-so-smart pricing recommendation.

---

## 📊 Example Output Snippet

**Input:**
* Product ID: `9286`
* Store ID: `16`
* Forecast Horizon: `7` days

**Output (Prepare to be amazed):**

Demand Forecast:

Expected Sales: 65.00 units over the next 7 days.
Inventory Status:

Current Stock: 700 units.
Status: Stock looks healthy compared to forecast demand (e.g., 700 vs. projected need of ~65 + buffer).
Action: No immediate reorder needed based on this short forecast. (Note: The original example showed a low stock trigger - adjust logic/thresholds in InventoryAgent as needed!)
Pricing Recommendation:

Rule-Based Check: [High stock (700) detected. Suggests potential discount.]
LLM Sentiment Analysis: [Negative feedback ('too expensive') found for product 9286.]
Combined Insight: [High stock + Negative sentiment = Prime candidate for a discount to move units & improve perception.]
Final Price: ₹30.18 (Example calculation based on increased discount)
Final Discount: 15.00% (Increased from baseline due to stock & sentiment)
*(Note: The exact output format and logic depend on your `app.py` and agent implementations. This is a representative example.)*

---

## 🎯 Why This Rocks (Benefits & Impact)

* **Say Goodbye to Stockouts**: Keep those shelves full (but not *too* full).
* **Slash Wasteful Overstock**: Stop burning cash on inventory just sitting there.
* **Price Like a Pro**: Maximize revenue with dynamic, data-driven pricing.
* **Supercharge Your Supply Chain**: Smooth, automated collaboration.
* **Happier Customers**: Pricing that reflects reality and responds to feedback.

---

## 🏆 Hackathon Domination

This project nails the hackathon requirements:

* ✅ **Multi-Agent Framework**: Check! (Thanks, CrewAI)
* ✅ **Ollama LLM Integration**: Check! (`gemma:2b` doing the heavy lifting)
* ✅ **Custom Tools**: Check! (Sentiment analysis, supplier comms)
* ✅ **SQLite Database**: Check!
* ✅ **Web Interface**: Check! (Flask frontend for easy interaction)

---

## 📹 Demo Video

[*(Click to watch the Demo)* 🎬](https://drive.google.com/file/d/1wwXp0G5wLFXceQHNNnmtRtl1axMK7J0g/view?usp=sharing)

---

## 📈 What's Next? (Future Bling)

We're not stopping here! Ideas for world domination include:

* **Real-Time Data Feeds**: Hook into live weather/event APIs for god-tier forecasting.
* **Dazzling Visualizations**: More charts! Everyone loves charts (Inventory trends, sales velocity).
* **Cloud Deployment**: Take this baby live on Heroku, AWS, or your favorite cloud.
* **Fort Knox Security**: User logins and roles for managers.
* **More Sophisticated Agents**: Agents that learn supplier lead times, analyze competitor pricing, etc.

---

## 👥 The Mastermind

* **[Somshubhro Guha]**: Lead Dev / AI Wrangler
* **[website]** : [somshubhro.com](https://somg-portfolio-1.onrender.com/)

---

## 📜 License

This project is under the MIT License - see the `LICENSE` file for the boring legal stuff.

---

## 🙏 Shoutouts

Big thanks to the creators and communities behind:

* **CrewAI**: Making multi-agent systems less painful.
* **Ollama**: Bringing powerful LLMs to our local machines.
* **Flask**: Keeping web development simple and fun.
* **SQLite**: The little database engine that could.

---

Now, go forth and optimize! 🚀