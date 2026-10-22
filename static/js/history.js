// Prediction History Controller
document.addEventListener('DOMContentLoaded', () => {
    const healthSelect = document.getElementById('filter-health');
    const classInput = document.getElementById('filter-class');
    const startDateInput = document.getElementById('filter-start-date');
    const endDateInput = document.getElementById('filter-end-date');
    const applyBtn = document.getElementById('btn-apply-filters');
    const tbody = document.getElementById('history-tbody');
    const totalCountEl = document.getElementById('history-total-count');

    async function loadHistory() {
        const health = healthSelect.value;
        const faultClass = classInput.value.trim();
        const start = startDateInput.value;
        const end = endDateInput.value;

        let query = '/api/predictions?limit=100';
        if (health) query += `&health_status=${encodeURIComponent(health)}`;
        if (faultClass) query += `&fault_class=${encodeURIComponent(faultClass)}`;
        if (start) query += `&start_date=${encodeURIComponent(start)}`;
        if (end) query += `&end_date=${encodeURIComponent(end)}`;

        try {
            const res = await fetch(query);
            const data = await res.json();

            if (data.success) {
                renderTable(data.predictions);
                totalCountEl.textContent = `Total: ${data.predictions.length} records`;
            }
        } catch (e) {
            console.error("Error loading prediction history:", e);
        }
    }

    function renderTable(predictions) {
        tbody.innerHTML = '';

        if (!predictions || predictions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No prediction history logs found matching selected criteria.</td></tr>';
            return;
        }

        predictions.forEach(item => {
            const tr = document.createElement('tr');

            const health = item.health_status.toUpperCase();
            const confidenceStr = `${parseFloat(item.confidence).toFixed(1)}%`;

            tr.innerHTML = `
                <td><code>#${item.id}</code></td>
                <td>${item.timestamp}</td>
                <td>${item.dataset_name}</td>
                <td><strong>${item.predicted_class}</strong></td>
                <td>${confidenceStr}</td>
                <td><span class="badge-status ${health}">${health}</span></td>
                <td><code style="font-size: 0.75rem;">${item.model_version || 'CNN-LSTM-v1'}</code></td>
            `;

            tbody.appendChild(tr);
        });
    }

    applyBtn.addEventListener('click', loadHistory);

    loadHistory();
});
