// Company Model Training Controller
document.addEventListener('DOMContentLoaded', () => {
    let accuracyChart = null;
    let lossChart = null;

    const form = document.getElementById('training-form');
    const startBtn = document.getElementById('btn-start-training');
    const statusBox = document.getElementById('training-status-box');
    const terminalLog = document.getElementById('terminal-log');

    async function initTrainingPage() {
        try {
            const res = await fetch('/api/inspect-current-dataset');
            const data = await res.json();
            if (data.success) {
                const ds = data.dataset;
                document.getElementById('train-ds-filename').value = `${ds.filename}${ds.is_demo ? ' (Demo)' : ''}`;

                // Populate Target Dropdown
                const targetSelect = document.getElementById('train-target-col');
                targetSelect.innerHTML = '<option value="">-- Select Target Fault Column --</option>';
                ds.columns.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    if (['fault_status', 'fault', 'condition', 'label', 'status'].includes(col.toLowerCase())) opt.selected = true;
                    targetSelect.appendChild(opt);
                });

                // Populate Machine ID Dropdown
                const machineSelect = document.getElementById('train-machine-id-col');
                machineSelect.innerHTML = '<option value="">None / Single Unit</option>';
                ds.columns.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    if (col.toLowerCase().includes('machine') || col.toLowerCase().includes('asset') || col.toLowerCase().includes('unit')) opt.selected = true;
                    machineSelect.appendChild(opt);
                });

                // Features list
                document.getElementById('train-features-list').textContent = ds.numeric_columns.join(', ');

                // Check stored dataset config from localStorage
                const storedConfigStr = localStorage.getItem('dataset_config');
                if (storedConfigStr) {
                    try {
                        const cfg = JSON.parse(storedConfigStr);
                        if (cfg.target_column) targetSelect.value = cfg.target_column;
                        if (cfg.machine_id_column) machineSelect.value = cfg.machine_id_column;
                        if (cfg.feature_columns && cfg.feature_columns.length > 0) {
                            document.getElementById('train-features-list').textContent = cfg.feature_columns.join(', ');
                        }
                        if (cfg.sequence_length) document.getElementById('train-seq-len').value = cfg.sequence_length;
                    } catch (e) {
                        console.error("Error reading dataset_config from localStorage:", e);
                    }
                }
            }
        } catch (e) {
            console.error("Error inspecting dataset for training:", e);
        }

        loadExistingTrainingMetrics();
    }

    async function loadExistingTrainingMetrics() {
        try {
            const res = await fetch('/api/training-history');
            const data = await res.json();

            if (data.success && data.trained && data.metrics) {
                logTerminal("[FOUND TRAINED MODEL] Loading saved evaluation metrics and training history...");
                renderEvaluationResults(data.metrics);
            } else {
                logTerminal("[INFO] No trained model found. Click 'Train CNN-LSTM Model' to execute training.");
            }
        } catch (e) {
            console.error("Error loading training history:", e);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const targetCol = document.getElementById('train-target-col').value;
        const machineIdCol = document.getElementById('train-machine-id-col').value;
        const seqLen = parseInt(document.getElementById('train-seq-len').value, 10);
        const epochs = parseInt(document.getElementById('train-epochs').value, 10);
        const batchSize = parseInt(document.getElementById('train-batch').value, 10);
        const lr = parseFloat(document.getElementById('train-lr').value);

        const featureStr = document.getElementById('train-features-list').textContent;
        const featureCols = featureStr.split(',').map(s => s.trim()).filter(s => s.length > 0);

        if (!targetCol) {
            alert("Please select a valid target column.");
            return;
        }

        startBtn.disabled = true;
        startBtn.innerHTML = '⏳ Training CNN-LSTM Model...';
        statusBox.textContent = 'Status: Training in progress... Please wait.';
        statusBox.style.color = 'var(--status-warning)';

        terminalLog.textContent = '';
        logTerminal(`[START] Initiating 1D CNN + LSTM Training Pipeline`);
        logTerminal(`[CONFIG] Target: '${targetCol}', Machine ID Col: '${machineIdCol || 'None'}', Seq Length: ${seqLen}, Epochs: ${epochs}, Batch Size: ${batchSize}, Learning Rate: ${lr}`);
        logTerminal(`[FEATURES] ${featureCols.join(', ')}`);
        logTerminal(`[PREPROCESSING] Cleaning duplicates, imputing missing values, and generating machine-boundary sequences...`);

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_column: targetCol,
                    feature_columns: featureCols,
                    sequence_length: seqLen,
                    epochs: epochs,
                    batch_size: batchSize,
                    learning_rate: lr,
                    machine_id_column: machineIdCol
                })
            });

            const data = await res.json();

            if (data.success) {
                logTerminal(`[SUCCESS] Model training completed successfully!`);
                logTerminal(`[EVALUATION] Test Accuracy: ${(data.metrics.test_accuracy * 100).toFixed(2)}%`);
                statusBox.textContent = 'Status: Training Complete!';
                statusBox.style.color = 'var(--status-normal)';

                renderEvaluationResults(data.metrics);
            } else {
                logTerminal(`[ERROR] ${data.error}`);
                statusBox.textContent = 'Status: Training Failed';
                statusBox.style.color = 'var(--status-critical)';
                alert(`Training Error: ${data.error}`);
            }
        } catch (err) {
            logTerminal(`[CRITICAL ERROR] ${err.message}`);
            statusBox.textContent = 'Status: Connection error';
            statusBox.style.color = 'var(--status-critical)';
        } finally {
            startBtn.disabled = false;
            startBtn.innerHTML = '⚡ Train CNN-LSTM Model';
        }
    });

    function logTerminal(msg) {
        const timeStr = new Date().toLocaleTimeString();
        terminalLog.textContent += `[${timeStr}] ${msg}\n`;
        terminalLog.scrollTop = terminalLog.scrollHeight;
    }

    function renderEvaluationResults(metrics) {
        document.getElementById('evaluation-results-section').style.display = 'block';

        document.getElementById('test-acc-val').textContent = `${(metrics.test_accuracy * 100).toFixed(2)}%`;
        document.getElementById('test-prec-val').textContent = `${(metrics.test_precision * 100).toFixed(2)}%`;
        document.getElementById('test-rec-val').textContent = `${(metrics.test_recall * 100).toFixed(2)}%`;
        document.getElementById('test-f1-val').textContent = `${(metrics.test_f1 * 100).toFixed(2)}%`;

        renderCurveChart(
            'accuracyChart',
            'Accuracy',
            metrics.history.accuracy.map(v => (v * 100).toFixed(2)),
            metrics.history.val_accuracy.map(v => (v * 100).toFixed(2)),
            accuracyChart,
            (c) => accuracyChart = c
        );

        renderCurveChart(
            'lossChart',
            'Loss',
            metrics.history.loss.map(v => v.toFixed(4)),
            metrics.history.val_loss.map(v => v.toFixed(4)),
            lossChart,
            (c) => lossChart = c
        );

        renderConfusionMatrix(metrics.class_names, metrics.confusion_matrix);
    }

    function renderCurveChart(canvasId, title, trainData, valData, chartInstance, setChartFn) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const epochs = Array.from({ length: trainData.length }, (_, i) => i + 1);

        if (chartInstance) chartInstance.destroy();

        const newChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: `Training ${title}`,
                        data: trainData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.1
                    },
                    {
                        label: `Validation ${title}`,
                        data: valData,
                        borderColor: '#10b981',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#f9fafb' } } },
                scales: {
                    x: { title: { display: true, text: 'Epoch', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: '#374151' } },
                    y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
                }
            }
        });

        setChartFn(newChart);
    }

    function renderConfusionMatrix(classNames, cm) {
        const thead = document.getElementById('cm-thead');
        const tbody = document.getElementById('cm-tbody');

        let headHtml = '<tr><th>Actual \\ Predicted</th>';
        classNames.forEach(c => { headHtml += `<th>${c}</th>`; });
        headHtml += '</tr>';
        thead.innerHTML = headHtml;

        let bodyHtml = '';
        cm.forEach((row, i) => {
            bodyHtml += `<tr><td><strong>${classNames[i]}</strong></td>`;
            row.forEach((val, j) => {
                const isDiagonal = (i === j);
                const bgStyle = isDiagonal ? 'background: rgba(16, 185, 129, 0.2); font-weight: bold; color: #10b981;' : '';
                bodyHtml += `<td style="${bgStyle}">${val}</td>`;
            });
            bodyHtml += '</tr>';
        });
        tbody.innerHTML = bodyHtml;
    }

    initTrainingPage();
});
