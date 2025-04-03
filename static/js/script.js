// static/js/script.js
document.getElementById('inputForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const product_id = document.getElementById('product_id').value;
    const store_id = document.getElementById('store_id').value;
    const forecast_horizon_days = document.getElementById('forecast_horizon_days').value;

    const response = await fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ product_id, store_id, forecast_horizon_days }),
    });

    if (response.ok) {
        const data = await response.json();
        document.getElementById('demand_forecast').textContent = data.demand_forecast;
        document.getElementById('inventory_status').textContent = data.inventory_status;
        document.getElementById('pricing_recommendation').textContent = data.pricing_recommendation;
        document.getElementById('final_price').textContent = data.final_price;
        document.getElementById('final_discount').textContent = data.final_discount;
    } else {
        const error = await response.json();
        alert('Error: ' + error.error);
    }
});

const ctx = document.getElementById('demandChart').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
        datasets: [{
            label: 'Demand Forecast',
            data: [50, 55, 60, 65, 70, 75, 80], // Mock data; replace with actual forecast
            borderColor: 'rgba(75, 192, 192, 1)',
            fill: false
        }]
    },
    options: {
        scales: { y: { beginAtZero: true } }
    }
});