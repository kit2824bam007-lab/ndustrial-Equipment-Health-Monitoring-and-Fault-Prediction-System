// System Reports Controller
document.addEventListener('DOMContentLoaded', async () => {
    async function loadReportData() {
        try {
            // 1. Fetch Dataset Summary
            const dsRes = await fetch('/api/inspect-current-dataset');
            const dsData = await dsRes.json();
            if (dsData.success) {
                const ds = dsData.dataset;
                document.getElementById('rep-ds-filename').textContent = ds.filename;
                document.getElementById('rep-ds-source').textContent = ds.is_demo ? 'CWRU Bearing Demo Dataset' : 'User Uploaded CSV';
                document.getElementById('rep-ds-rows').textContent = ds.total_rows.toLocaleString();
                document.getElementById('rep-ds-sensors').textContent = ds.numeric_columns.join(', ');
            }

            // 2. Fetch Model Evaluation Info
            const trainRes = await fetch('/api/training-history');
            const trainData = await trainRes.json();
            if (trainData.success && trainData.trained && trainData.metrics) {
                const m = trainData.metrics;
                document.getElementById('rep-ds-seq').textContent = `${m.sequence_length || 100} time steps per window`;
                document.getElementById('rep-model-acc').textContent = `${(m.test_accuracy * 100).toFixed(2)}%`;
                document.getElementById('rep-model-prec').textContent = `${(m.test_precision * 100).toFixed(2)}%`;
                document.getElementById('rep-model-rec').textContent = `${(m.test_recall * 100).toFixed(2)}%`;
                document.getElementById('rep-model-f1').textContent = `${(m.test_f1 * 100).toFixed(2)}%`;
            } else {
                document.getElementById('rep-ds-seq').textContent = '100 time steps';
                document.getElementById('rep-model-acc').textContent = 'Model not trained';
                document.getElementById('rep-model-prec').textContent = '--';
                document.getElementById('rep-model-rec').textContent = '--';
                document.getElementById('rep-model-f1').textContent = '--';
            }

            // 3. Fetch Prediction Stats
            const predRes = await fetch('/api/predictions');
            const predData = await predRes.json();
            if (predData.success) {
                const stats = predData.stats;
                const hc = stats.health_counts || {};
                document.getElementById('rep-total-preds').textContent = stats.total_predictions || 0;
                document.getElementById('rep-normal-preds').textContent = hc.NORMAL || 0;
                document.getElementById('rep-warning-preds').textContent = hc.WARNING || 0;
                document.getElementById('rep-critical-preds').textContent = hc.CRITICAL || 0;
            }

        } catch (e) {
            console.error("Error loading report summary:", e);
        }
    }

    await loadReportData();
});
