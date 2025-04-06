// static/js/script.js
document.getElementById('inputForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const resultsSection = document.getElementById('results');
    resultsSection.classList.add('hidden');

    const product_id = document.getElementById('product_id').value;
    const store_id = document.getElementById('store_id').value;
    const forecast_horizon_days = parseInt(document.getElementById('forecast_horizon_days').value);

    const response = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_id, store_id, forecast_horizon_days }),
    });

    if (response.ok) {
        const data = await response.json();
        document.getElementById('demand_forecast').textContent = data.demand_forecast;
        document.getElementById('inventory_status').textContent = data.inventory_status;
        document.getElementById('order_quantity').textContent = data.order_quantity;
        document.getElementById('pricing_recommendation').textContent = data.pricing_recommendation;
        document.getElementById('final_price').textContent = data.final_price;
        document.getElementById('final_discount').textContent = data.final_discount;

        // Update chart
        const demandChart = Chart.getChart('demandChart');
        demandChart.data.labels = Array.from({ length: forecast_horizon_days }, (_, i) => `Day ${i + 1}`);
        demandChart.data.datasets[0].data = data.forecast_data;
        demandChart.update();

        resultsSection.classList.remove('hidden');
    } else {
        const error = await response.json();
        alert('Error: ' + error.error);
    }
});