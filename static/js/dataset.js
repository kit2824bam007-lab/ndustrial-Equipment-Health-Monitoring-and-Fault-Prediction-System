// Company Dataset Controller
document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const prepareBtn = document.getElementById('btn-prepare-dataset');

    // Drag and Drop handlers
    ['dragenter', 'dragover'].forEach(name => {
        uploadZone.addEventListener(name, (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        uploadZone.addEventListener(name, (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });
    });

    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileUpload(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFileUpload(fileInput.files[0]);
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith('.csv')) {
            uploadStatus.innerHTML = '<span style="color: var(--status-critical);">Error: File must be a valid .csv dataset.</span>';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        uploadStatus.innerHTML = '<span style="color: var(--primary);">Uploading and inspecting company dataset...</span>';

        try {
            const res = await fetch('/api/upload-dataset', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.success) {
                uploadStatus.innerHTML = '<span style="color: var(--status-normal);">✅ File uploaded and inspected successfully!</span>';
                renderDatasetInfo(data.dataset);
                if (window.updateActiveDatasetPill) window.updateActiveDatasetPill();
            } else {
                uploadStatus.innerHTML = `<span style="color: var(--status-critical);">Error: ${data.error}</span>`;
            }
        } catch (e) {
            uploadStatus.innerHTML = `<span style="color: var(--status-critical);">Upload failed: ${e.message}</span>`;
        }
    }

    async function loadCurrentDataset() {
        try {
            const res = await fetch('/api/inspect-current-dataset');
            const data = await res.json();
            if (data.success) renderDatasetInfo(data.dataset);
        } catch (e) {
            console.error("Error loading current dataset info:", e);
        }
    }

    window.loadCompanySampleDataset = async function() {
        try {
            const res = await fetch('/api/use-company-sample', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                uploadStatus.innerHTML = '<span style="color: var(--status-normal);">Switched to company_machine_data.csv baseline dataset.</span>';
                renderDatasetInfo(data.dataset);
                if (window.updateActiveDatasetPill) window.updateActiveDatasetPill();
            }
        } catch (e) {
            console.error("Error setting sample dataset:", e);
        }
    };

    function renderDatasetInfo(ds) {
        document.getElementById('ds-filename').textContent = `${ds.filename}${ds.is_demo ? ' (Demo)' : ''}`;
        document.getElementById('ds-filesize').textContent = ds.file_size || '--';
        document.getElementById('ds-rows').textContent = ds.total_rows.toLocaleString();
        document.getElementById('ds-cols').textContent = ds.total_columns;
        document.getElementById('ds-duplicates').textContent = ds.duplicate_count || 0;

        const missingCount = Object.values(ds.missing_counts || {}).reduce((a, b) => a + b, 0);
        document.getElementById('ds-missing').textContent = missingCount === 0 ? '0 (Clean)' : `${missingCount} missing values detected`;

        // 1. Timestamp Dropdown
        const timeSelect = document.getElementById('timestamp-col-select');
        timeSelect.innerHTML = '<option value="">-- None / Index Order --</option>';
        ds.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col.toLowerCase().includes('time') || col.toLowerCase().includes('date')) opt.selected = true;
            timeSelect.appendChild(opt);
        });

        // 2. Machine ID Dropdown
        const machineSelect = document.getElementById('machine-id-col-select');
        machineSelect.innerHTML = '<option value="">-- Single Equipment Unit --</option>';
        ds.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col.toLowerCase().includes('machine') || col.toLowerCase().includes('asset') || col.toLowerCase().includes('unit')) opt.selected = true;
            machineSelect.appendChild(opt);
        });

        // 3. Target Dropdown
        const targetSelect = document.getElementById('target-col-select');
        targetSelect.innerHTML = '<option value="">-- Select Target Column --</option>';
        ds.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (['fault_status', 'fault', 'condition', 'label', 'status', 'failure_status'].includes(col.toLowerCase())) opt.selected = true;
            targetSelect.appendChild(opt);
        });

        // 4. Sensor Feature Checkboxes
        const sensorContainer = document.getElementById('sensor-checkbox-group');
        sensorContainer.innerHTML = '';
        ds.numeric_columns.forEach(col => {
            const label = document.createElement('label');
            label.style.display = 'flex';
            label.style.alignItems = 'center';
            label.style.gap = '0.35rem';
            label.style.cursor = 'pointer';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'sensor_column';
            checkbox.value = col;
            checkbox.checked = true;

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(col));
            sensorContainer.appendChild(label);
        });

        renderTablePreview(ds.preview);
    }

    function renderTablePreview(previewList) {
        if (!previewList || previewList.length === 0) return;

        const cols = Object.keys(previewList[0]);
        const trHead = document.getElementById('preview-thead-tr');
        trHead.innerHTML = '';
        cols.forEach(c => {
            const th = document.createElement('th');
            th.textContent = c;
            trHead.appendChild(th);
        });

        const tbody = document.getElementById('preview-tbody');
        tbody.innerHTML = '';
        previewList.forEach(row => {
            const tr = document.createElement('tr');
            cols.forEach(c => {
                const td = document.createElement('td');
                td.textContent = row[c] !== null ? row[c] : 'NaN';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    // PREPARE DATASET Button Handler
    prepareBtn.addEventListener('click', async () => {
        const targetCol = document.getElementById('target-col-select').value;
        const timeCol = document.getElementById('timestamp-col-select').value;
        const machineIdCol = document.getElementById('machine-id-col-select').value;
        const seqLen = parseInt(document.getElementById('seq-len-input').value, 10);
        const valSize = parseFloat(document.getElementById('val-size-input').value);
        const testSize = parseFloat(document.getElementById('test-size-input').value);

        const featureCheckboxes = document.querySelectorAll('input[name="sensor_column"]:checked');
        const featureCols = Array.from(featureCheckboxes).map(cb => cb.value);

        if (!targetCol) {
            alert("Please select a valid target column.");
            return;
        }

        if (featureCols.length === 0) {
            alert("No suitable numeric sensor columns were selected.");
            return;
        }

        prepareBtn.disabled = true;
        prepareBtn.innerHTML = '⏳ Preparing Dataset...';

        try {
            const res = await fetch('/api/prepare-dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_column: targetCol,
                    feature_columns: featureCols,
                    sequence_length: seqLen,
                    test_size: testSize,
                    val_size: valSize,
                    timestamp_column: timeCol,
                    machine_id_column: machineIdCol
                })
            });

            const data = await res.json();

            if (data.success) {
                const prep = data.preparation;
                document.getElementById('prep-summary-container').style.display = 'block';

                document.getElementById('prep-seq-count').textContent = prep.total_sequences.toLocaleString();
                document.getElementById('prep-train-count').textContent = `${prep.train_samples.toLocaleString()} (${prep.train_ratio_pct}%)`;
                document.getElementById('prep-val-count').textContent = `${prep.val_samples.toLocaleString()} (${prep.val_ratio_pct}%)`;
                document.getElementById('prep-test-count').textContent = `${prep.test_samples.toLocaleString()} (${prep.test_ratio_pct}%)`;

                document.getElementById('prep-class-num').textContent = prep.num_classes;
                document.getElementById('prep-class-names').textContent = prep.class_names.join(', ');

                const distStr = Object.entries(prep.class_distribution).map(([k, v]) => `${k}: ${v}`).join(' | ');
                document.getElementById('prep-class-dist').textContent = distStr;

                // Save selected config to localStorage for seamless sync with Training page
                localStorage.setItem('dataset_config', JSON.stringify({
                    target_column: targetCol,
                    feature_columns: featureCols,
                    sequence_length: seqLen,
                    test_size: testSize,
                    val_size: valSize,
                    timestamp_column: timeCol,
                    machine_id_column: machineIdCol
                }));

            } else {
                alert(`Dataset Preparation Error: ${data.error}`);
            }
        } catch (e) {
            alert(`Failed to prepare dataset: ${e.message}`);
        } finally {
            prepareBtn.disabled = false;
            prepareBtn.innerHTML = '⚙️ PREPARE DATASET';
        }
    });

    loadCurrentDataset();
});
