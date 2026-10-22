// Company Machine Health Prediction Controller
document.addEventListener('DOMContentLoaded', () => {
    let probChart = null;
    let sensorChart = null;

    const runBtn = document.getElementById('btn-run-predict');
    const fileInput = document.getElementById('predict-file-input');
    const resultContainer = document.getElementById('prediction-result-container');

    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true;
        runBtn.innerHTML = '⏳ Predicting Machine Health...';

        const formData = new FormData();
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
        }

        try {
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.success) {
                renderPredictionResults(data.prediction);
            } else {
                resultContainer.style.display = 'block';
                document.getElementById('pred-alert-box').innerHTML = `
                    <div class="alert-banner critical">
                        <span style="font-size: 1.5rem;">⚠️</span>
                        <div>
                            <div class="alert-title">Prediction Error</div>
                            <div class="alert-desc">${data.error}</div>
                        </div>
                    </div>
                `;
            }
        } catch (err) {
            alert(`Prediction execution failed: ${err.message}`);
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = '⚡ Predict Machine Health';
        }
    });

    function renderPredictionResults(pred) {
        resultContainer.style.display = 'block';

        // 1. Alert Banner
        const alertBox = document.getElementById('pred-alert-box');
        const alertObj = pred.alert || {};
        const levelClass = alertObj.level || 'normal';

        alertBox.innerHTML = `
            <div class="alert-banner ${levelClass}">
                <span style="font-size: 1.5rem;">${levelClass === 'normal' ? '✅' : '⚠️'}</span>
                <div>
                    <div class="alert-title">${alertObj.title || 'Prediction Complete'}</div>
                    <div class="alert-desc">${alertObj.message || ''}</div>
                </div>
            </div>
        `;

        // 2. Summary KPI Cards
        document.getElementById('res-machine-id').textContent = pred.machine_id || 'M001';
        document.getElementById('res-timestamp').textContent = pred.timestamp || 'N/A';
        document.getElementById('res-predicted-class').textContent = pred.predicted_class;
        document.getElementById('res-dataset-name').textContent = pred.dataset_name;
        document.getElementById('res-confidence').textContent = `${pred.confidence}%`;
        
        // Health Score (0-100) & Status Badge
        const score = pred.health_score !== undefined ? pred.health_score : 100.0;
        document.getElementById('res-health-score').textContent = `${score}/100`;

        const hs = (pred.health_status || 'NORMAL').toUpperCase();
        document.getElementById('res-health-status').innerHTML = `<span class="badge-status ${hs}">${hs}</span>`;

        // 3. Charts
        renderProbabilityChart(pred.probabilities);
        renderSensorTrendChart(pred.sensor_trends);
    }

    function renderProbabilityChart(probDict) {
        const ctx = document.getElementById('predProbChart').getContext('2d');
        const labels = Object.keys(probDict);
        const values = Object.values(probDict);

        if (probChart) probChart.destroy();

        probChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: values,
                    backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' }, min: 0, max: 100 },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
                }
            }
        });
    }

    function renderSensorTrendChart(sensorTrends) {
        const ctx = document.getElementById('predSensorChart').getContext('2d');
        const keys = Object.keys(sensorTrends);

        if (keys.length === 0) return;

        const firstKey = keys[0];
        const dataPoints = sensorTrends[firstKey] || [];
        const labels = Array.from({ length: dataPoints.length }, (_, i) => i + 1);

        if (sensorChart) sensorChart.destroy();

        sensorChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: firstKey,
                    data: dataPoints,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#f9fafb' } } },
                scales: {
                    x: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
                }
            }
        });
    }
});
