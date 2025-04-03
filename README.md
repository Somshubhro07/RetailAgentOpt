# Retail Inventory Optimizer

**Optimizing Retail Inventory with a Multi-Agent AI System**

Welcome to **Retail Inventory Optimizer**, a cutting-edge multi-agent AI system designed to transform retail inventory management. Developed for the **Gen AI Hackathon**, this project tackles the challenge of balancing product availability and inventory costs by leveraging AI agents, on-premise LLMs (Ollama), and an intuitive web interface. Our solution predicts demand, manages stock levels, optimizes pricing, and enhances supply chain efficiency—all while fostering seamless collaboration between stores, warehouses, suppliers, and customers.

---

## 📋 Project Overview

### **Problem Statement**
Retail chains face significant challenges in maintaining an optimal balance between product availability and inventory costs:
- **Stockouts**: Running out of popular items, leading to lost sales.
- **Overstocking**: Excess inventory, increasing holding costs.
- **Inefficient Pricing**: Suboptimal pricing strategies for slow-moving inventory.
- **Supply Chain Delays**: Lack of proactive collaboration with suppliers.

### **Our Solution**
Retail Inventory Optimizer employs a multi-agent framework where specialized AI agents collaborate to manage inventory proactively:
- **DemandForecastAgent**: Predicts future demand using historical sales and external factors (e.g., holidays, weather).
- **InventoryAgent**: Monitors stock levels and collaborates with the SupplierAgent to reorder products when needed.
- **PricingAgent**: Recommends optimal prices and discounts by analyzing demand, inventory, and customer sentiment.
- **SupplierAgent**: Automates reordering to prevent stockouts and streamline supply chain operations.

**Key Technologies:**
- **Ollama LLM (gemma:2b)**: For demand forecasting refinement, pricing recommendations, and sentiment analysis.
- **SQLite Database**: Stores sales, inventory, customer feedback, and external factors.
- **Flask Web App**: Provides a user-friendly interface to interact with the system.

---

## 🚀 Features
- **Demand Forecasting**: Predicts sales for a given product and store over a specified horizon, considering external factors like holidays and weather.
- **Inventory Management**: Monitors stock levels, identifies low stock, and automatically places orders with suppliers.
- **Dynamic Pricing**: Recommends prices and discounts based on demand, inventory, and customer sentiment analysis.
- **Supplier Collaboration**: Proactively reorders products to prevent stockouts, improving supply chain efficiency.
- **Customer Sentiment Analysis**: Analyzes feedback to adjust pricing strategies (e.g., increasing discounts for negative sentiment).
- **Web Interface**: A responsive Flask-based web app for users to input product/store details and view agent outputs.

---

## 🛠️ Tech Stack

### **Backend**
- Python 3.8+
- Flask (Web Framework)
- SQLite (Database)
- Ollama LLM (gemma:2b) for on-premise AI
- CrewAI (Multi-Agent Framework)

### **Frontend**
- HTML/CSS/JavaScript
- Chart.js (for visualizations, optional)

### **Tools**
- LangChain-Ollama (for LLM integration)
- SQLite3 (for database operations)

---

## 📂 Project Structure
```
RetailAgentopt/
├── agents.py                 # Agent definitions (DemandForecastAgent, InventoryAgent, PricingAgent, SupplierAgent)
├── sentiment_analyzer.py     # Sentiment analysis using Ollama LLM
├── app.py                    # Flask backend for the web app
├── ecommerce_data.sqlite     # SQLite database (sales, inventory, feedback, external factors)
├── templates/
│   ├── index.html            # Main HTML template for the web app
│   └── style.css             # CSS for styling
├── static/
│   ├── js/
│   │   └── script.js         # JavaScript for frontend interactivity
│   └── css/
│       └── style.css         # CSS (if separate)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🖥️ Setup Instructions

### **Prerequisites**
- Python 3.8 or higher
- SQLite
- Ollama installed and running (gemma:2b model)
- Google Chrome (for Chart.js compatibility)

### **Installation**
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/RetailAgentopt.git
    cd RetailAgentopt
    ```
2. **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Set Up Ollama:**
    - Ensure Ollama is running on `http://localhost:11434`.
    - Install the gemma:2b model:
      ```bash
      ollama pull gemma:2b
      ```
5. **Initialize the Database:**
    - If the `ecommerce_data.sqlite` database is missing, create and populate it:
      ```sql
      sqlite3 ecommerce_data.sqlite
      CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id TEXT, store_id TEXT, sales INTEGER, date TEXT);
      CREATE TABLE inventory (id INTEGER PRIMARY KEY, product_id TEXT, store_id TEXT, stock_level INTEGER);
      CREATE TABLE customer_feedback (id INTEGER PRIMARY KEY, product_id TEXT, feedback TEXT, sentiment TEXT, timestamp TEXT);
      CREATE TABLE external_factors (id INTEGER PRIMARY KEY, date TEXT, factor_type TEXT, factor_value TEXT, impact REAL);
      INSERT INTO sales (product_id, store_id, sales, date) VALUES ('9286', '16', 50, '2025-04-01');
      INSERT INTO inventory (product_id, store_id, stock_level) VALUES ('9286', '16', 700);
      INSERT INTO customer_feedback (product_id, feedback, sentiment, timestamp) VALUES ('9286', 'This Bluetooth Speaker is too expensive for its quality.', 'negative', '2025-04-01');
      INSERT INTO external_factors (date, factor_type, factor_value, impact) VALUES ('2025-04-01', 'holiday', 'Diwali', 1.5);
      ```
6. **Run the Web App:**
    ```bash
    python app.py
    ```
7. **Access the Web App:**
    Open your browser at `http://127.0.0.1:5000/`.

---

## 🌐 Usage
1. **Navigate to the Web App:**
    Visit `http://127.0.0.1:5000/` in your browser.
2. **Input Details:**
    Enter `product_id` (e.g., 9286), `store_id` (e.g., 16), and `forecast_horizon_days` (e.g., 7).
3. **View Results:**
    - **Demand Forecast**: Expected sales over the specified horizon.
    - **Inventory Status**: Current stock level and any reordering actions.
    - **Pricing Recommendation**: Suggested price, discount, and reasoning.

---

## 📊 Example Output
**Input:**
- Product ID: 9286
- Store ID: 16
- Forecast Horizon: 7 days

**Output:**
- **Demand Forecast**: Expected sales: 65.00 units over 7 days.
- **Inventory Status**: Stock low (700 < 97.5). Order placed: 50 units.
- **Pricing Recommendation**: 
  - Rule Action: [High stock (700) with Stable demand. Increased discount to 10.0%].
  - LLM Analysis: [The rule-based suggestion of 31.95 might not consider sentiment trends.].
  - Sentiment Insight: [High negative sentiment detected; Increased discount to 15.0%].
  - **Final Price**: ₹30.18
  - **Final Discount**: 15.00%

---

## 🎯 Benefits and Impact
- **Reduced Stockouts**: Proactive demand forecasting and supplier collaboration ensure product availability.
- **Minimized Overstock**: Inventory monitoring prevents excess stock, reducing holding costs.
- **Optimized Pricing**: Dynamic pricing based on demand, inventory, and customer sentiment maximizes sales.
- **Improved Supply Chain**: Automated reordering streamlines operations between stores and suppliers.
- **Enhanced Customer Satisfaction**: Sentiment-driven pricing adjustments address customer feedback.

---

## 🏆 Hackathon Alignment
This project meets all hackathon requirements:
- **Multi-Agent Framework**: CrewAI-based agents (DemandForecastAgent, InventoryAgent, PricingAgent, SupplierAgent).
- **Ollama LLM**: Used for demand forecasting, pricing recommendations, and sentiment analysis.
- **Custom Tools**: Sentiment analysis, supplier collaboration, and external factor integration.
- **SQLite DB**: Stores sales, inventory, feedback, and external factors.
- **Web Interface**: Flask-based frontend for user interaction.

---

## 📹 Demo Video
A demo video showcasing the web app in action will be linked here: *(to be added after recording).*

---

## 📈 Future Improvements
- **Real-Time Data Integration**: Add APIs for weather or holiday data to enhance demand forecasting.
- **Advanced Visualizations**: Integrate more charts (e.g., inventory trends, sales history).
- **Scalability**: Deploy on a cloud platform like Heroku or AWS for production use.
- **User Authentication**: Add login functionality for retail managers.

---

## 👥 Team
- **[Your Name]**: Lead Developer, AI Engineer  
*(Add other team members if applicable)*

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- **CrewAI**: For the multi-agent framework.
- **Ollama**: For providing on-premise LLMs.
- **Flask**: For the lightweight web framework.
- **Chart.js**: For visualization (optional).

Let’s optimize retail inventory together! 🚀