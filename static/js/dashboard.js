// Dashboard Controller
document.addEventListener('DOMContentLoaded', async () => {
    let sensorTrendChart = null;
    let predDistChart = null;

    async function loadDashboardSummary() {
        try {
            const res = await fetch('/api/dashboard-summary');
            const data = await res.json();

            if (!data.success) return;

            // 1. Update KPI Cards
            const healthValEl = document.getElementById('card-health-status');
            const predictedFaultEl = document.getElementById('card-predicted-fault');
            const confidenceEl = document.getElementById('card-confidence');
            const accuracyEl = document.getElementById('card-model-accuracy');
            const alertContainer = document.getElementById('dashboard-alert-container');

            // Model Accuracy Card
            if (data.model_trained && data.test_accuracy !== null) {
                accuracyEl.textContent = `${data.test_accuracy}%`;
                accuracyEl.style.color = '#3b82f6';
            } else {
                accuracyEl.textContent = 'Model not trained';
                accuracyEl.style.color = 'var(--text-muted)';
            }

            // Latest Prediction KPI & Alert
            if (data.latest_prediction) {
                const lp = data.latest_prediction;
                predictedFaultEl.textContent = lp.predicted_class;
                confidenceEl.textContent = `${parseFloat(lp.confidence).toFixed(1)}%`;

                // Health Badge
                const health = lp.health_status.toUpperCase();
                healthValEl.innerHTML = `<span class="badge-status ${health}">${health}</span>`;

                // Alert Banner
                if (health === 'CRITICAL') {
                    alertContainer.innerHTML = `
                        <div class="alert-banner critical">
                            <span style="font-size: 1.5rem;">⚠️</span>
                            <div>
                                <div class="alert-title">CRITICAL ALERT - Equipment Fault Condition Detected</div>
                                <div class="alert-desc">Machine predicted fault state: '<strong>${lp.predicted_class}</strong>' (${parseFloat(lp.confidence).toFixed(1)}% confidence). Inspect equipment immediately.</div>
                            </div>
                        </div>
                    `;
                } else if (health === 'WARNING') {
                    alertContainer.innerHTML = `
                        <div class="alert-banner warning">
                            <span style="font-size: 1.5rem;">⚠️</span>
                            <div>
                                <div class="alert-title">WARNING - Abnormal Equipment Behavior</div>
                                <div class="alert-desc">Machine showing degrading condition: '<strong>${lp.predicted_class}</strong>' (${parseFloat(lp.confidence).toFixed(1)}% confidence). Monitor parameters closely.</div>
                            </div>
                        </div>
                    `;
                } else {
                    alertContainer.innerHTML = `
                        <div class="alert-banner normal">
                            <span style="font-size: 1.5rem;">✅</span>
                            <div>
                                <div class="alert-title">SYSTEM HEALTHY - Normal Operating Parameters</div>
                                <div class="alert-desc">Equipment operating normally. Machine condition: '<strong>${lp.predicted_class}</strong>'.</div>
                            </div>
                        </div>
                    `;
                }
            } else {
                healthValEl.innerHTML = `<span class="badge-status NORMAL">NO PREDICTIONS</span>`;
                predictedFaultEl.textContent = 'No data';
                confidenceEl.textContent = '--';
                alertContainer.innerHTML = '';
            }

            // 2. Render Prediction Class Distribution Chart
            renderClassDistributionChart(data.stats.class_counts || {});

            // 3. Sensor Signals Selector & Chart
            setupSensorTrends(data.sensor_trends || {});

        } catch (e) {
            console.error("Error loading dashboard summary:", e);
        }
    }

    function renderClassDistributionChart(classCounts) {
        const ctx = document.getElementById('predictionDistChart').getContext('2d');
        const labels = Object.keys(classCounts);
        const values = Object.values(classCounts);

        if (labels.length === 0) {
            labels.push('Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault');
            values.push(0, 0, 0, 0);
        }

        if (predDistChart) predDistChart.destroy();

        predDistChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Count',
                    data: values,
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                    y: { ticks: { color: '#9ca3af', stepSize: 1 }, grid: { color: '#374151' }, beginAtZero: true }
                }
            }
        });
    }

    function setupSensorTrends(sensorTrends) {
        const selector = document.getElementById('sensor-selector');
        selector.innerHTML = '';

        const keys = Object.keys(sensorTrends);
        if (keys.length === 0) {
            selector.innerHTML = '<option value="">No sensor signals available</option>';
            return;
        }

        keys.forEach(k => {
            const opt = document.createElement('option');
            opt.value = k;
            opt.textContent = k;
            selector.appendChild(opt);
        });

        selector.addEventListener('change', () => {
            renderSensorTrendChart(selector.value, sensorTrends[selector.value] || []);
        });

        // Initial render first sensor
        renderSensorTrendChart(keys[0], sensorTrends[keys[0]] || []);
    }

    function renderSensorTrendChart(sensorName, dataPoints) {
        const ctx = document.getElementById('sensorTrendChart').getContext('2d');
        const labels = Array.from({ length: dataPoints.length }, (_, i) => i + 1);

        if (sensorTrendChart) sensorTrendChart.destroy();

        sensorTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: sensorName,
                    data: dataPoints,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#f9fafb' } }
                },
                scales: {
                    x: { title: { display: true, text: 'Time Step / Sample Index', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                    y: { title: { display: true, text: 'Signal Magnitude', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
                }
            }
        });
    }

    await loadDashboardSummary();
});
