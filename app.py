import os
import json
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
from utils.database import (
    init_db, get_predictions, get_latest_prediction,
    get_latest_training_run, get_prediction_stats, get_alert_history
)
from utils.helpers import allowed_file, format_file_size
from data_loader import inspect_company_csv
from models.preprocessing import preview_dataset_preparation
from models.train import run_company_training_pipeline
from models.predict import predict_machine_health

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Initialize SQLite database schema
init_db()

# Global state tracking current active dataset
DEFAULT_COMPANY_CSV = os.path.join(config.SAMPLE_DIR, 'company_machine_data.csv')
CURRENT_DATASET = {
    'filepath': DEFAULT_COMPANY_CSV if os.path.exists(DEFAULT_COMPANY_CSV) else os.path.join(config.SAMPLE_DIR, 'sample_machine_data.csv'),
    'filename': 'company_machine_data.csv' if os.path.exists(DEFAULT_COMPANY_CSV) else 'sample_machine_data.csv',
    'is_demo': True
}

# ==================== PAGE ROUTES ====================

@app.route('/')
@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html', active_page='dataset')

@app.route('/training')
def training_page():
    return render_template('training.html', active_page='training')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html', active_page='prediction')

@app.route('/history')
def history_page():
    return render_template('history.html', active_page='history')

@app.route('/reports')
def reports_page():
    return render_template('reports.html', active_page='reports')

@app.route('/about')
def about_page():
    return render_template('about.html', active_page='about')

# ==================== API ENDPOINTS ====================

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Endpoint for uploading real company machine dataset CSV files."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part attached in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only standard CSV files (.csv) are permitted.'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(config.UPLOADS_DIR, f"{int(datetime.now().timestamp())}_{filename}")
    file.save(save_path)

    try:
        inspection = inspect_company_csv(save_path)
        CURRENT_DATASET['filepath'] = save_path
        CURRENT_DATASET['filename'] = filename
        CURRENT_DATASET['is_demo'] = False

        inspection['is_demo'] = False
        inspection['file_size'] = format_file_size(os.path.getsize(save_path))
        return jsonify({'success': True, 'dataset': inspection})
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({'error': f"Failed to inspect company CSV dataset: {str(e)}"}), 400

@app.route('/api/inspect-current-dataset', methods=['GET'])
def inspect_current():
    """Inspect currently active company dataset."""
    filepath = CURRENT_DATASET['filepath']
    if not os.path.exists(filepath):
        filepath = DEFAULT_COMPANY_CSV if os.path.exists(DEFAULT_COMPANY_CSV) else os.path.join(config.SAMPLE_DIR, 'sample_machine_data.csv')
        CURRENT_DATASET['filepath'] = filepath
        CURRENT_DATASET['filename'] = os.path.basename(filepath)
        CURRENT_DATASET['is_demo'] = True

    try:
        inspection = inspect_company_csv(filepath)
        inspection['is_demo'] = CURRENT_DATASET.get('is_demo', False)
        inspection['file_size'] = format_file_size(os.path.getsize(filepath))
        return jsonify({'success': True, 'dataset': inspection})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/use-company-sample', methods=['POST'])
def use_company_sample():
    """Switch active dataset to baseline company_machine_data.csv sample demo dataset."""
    filepath = DEFAULT_COMPANY_CSV
    if not os.path.exists(filepath):
        import create_company_sample_data

    CURRENT_DATASET['filepath'] = filepath
    CURRENT_DATASET['filename'] = 'company_machine_data.csv'
    CURRENT_DATASET['is_demo'] = True

    inspection = inspect_company_csv(filepath)
    inspection['is_demo'] = True
    inspection['file_size'] = format_file_size(os.path.getsize(filepath))
    return jsonify({'success': True, 'dataset': inspection})

@app.route('/api/prepare-dataset', methods=['POST'])
def prepare_dataset_summary():
    """Preview 3D sequence window counts, class distribution, and split sizes before training."""
    data = request.json or {}
    target_col = data.get('target_column')
    feature_cols = data.get('feature_columns', [])
    seq_len = int(data.get('sequence_length', 30))
    test_size = float(data.get('test_size', 0.15))
    val_size = float(data.get('val_size', 0.15))
    timestamp_col = data.get('timestamp_column')
    machine_id_col = data.get('machine_id_column')

    filepath = CURRENT_DATASET.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No dataset selected.'}), 400

    if not target_col:
        return jsonify({'error': 'Please select a valid target column.'}), 400

    if not feature_cols:
        return jsonify({'error': 'No suitable numeric sensor columns were found or selected.'}), 400

    try:
        summary = preview_dataset_preparation(
            file_path=filepath,
            target_column=target_col,
            feature_columns=feature_cols,
            sequence_length=seq_len,
            test_size=test_size,
            val_size=val_size,
            machine_id_column=machine_id_col,
            timestamp_column=timestamp_col
        )
        return jsonify({'success': True, 'preparation': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    """Trigger CNN-LSTM model training on company dataset."""
    data = request.json or {}
    target_col = data.get('target_column')
    feature_cols = data.get('feature_columns', [])
    seq_len = int(data.get('sequence_length', 30))
    test_size = float(data.get('test_size', 0.15))
    val_size = float(data.get('val_size', 0.15))
    epochs = int(data.get('epochs', config.DEFAULT_EPOCHS))
    batch_size = int(data.get('batch_size', config.DEFAULT_BATCH_SIZE))
    learning_rate = float(data.get('learning_rate', 0.001))
    timestamp_col = data.get('timestamp_column')
    machine_id_col = data.get('machine_id_column')

    filepath = CURRENT_DATASET.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No dataset selected for training.'}), 400

    try:
        metrics = run_company_training_pipeline(
            file_path=filepath,
            target_column=target_col,
            feature_columns=feature_cols,
            sequence_length=seq_len,
            test_size=test_size,
            val_size=val_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            timestamp_column=timestamp_col,
            machine_id_column=machine_id_col
        )
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Execute prediction inference on unseen machine data."""
    file_path = None

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid CSV dataset format.'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(config.UPLOADS_DIR, f"predict_{int(datetime.now().timestamp())}_{filename}")
        file.save(file_path)
    else:
        file_path = CURRENT_DATASET.get('filepath')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No input dataset provided for prediction.'}), 400

    try:
        result = predict_machine_health(file_path)
        return jsonify({'success': True, 'prediction': result})
    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'not_trained': True}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/api/training-history', methods=['GET'])
def get_training_metrics():
    """Retrieve saved model training metrics and evaluation curves."""
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify({'success': True, 'trained': True, 'metrics': metrics})
    else:
        return jsonify({'success': True, 'trained': False, 'message': 'No trained model found.'})

@app.route('/api/predictions', methods=['GET'])
def list_predictions():
    """Retrieve filterable prediction logs from SQLite database."""
    fault_class = request.args.get('fault_class')
    health_status = request.args.get('health_status')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = int(request.args.get('limit', 100))

    logs = get_predictions(limit=limit, fault_class=fault_class, health_status=health_status, start_date=start_date, end_date=end_date)
    stats = get_prediction_stats()
    return jsonify({'success': True, 'predictions': logs, 'stats': stats})

@app.route('/api/alerts', methods=['GET'])
def list_alerts():
    """Retrieve warning and critical alert history from SQLite database."""
    alerts = get_alert_history(limit=100)
    return jsonify({'success': True, 'alerts': alerts})

@app.route('/api/dashboard-summary', methods=['GET'])
def dashboard_summary():
    """Provide aggregated real-time monitoring stats for Dashboard overview."""
    latest_pred = get_latest_prediction()
    
    # Model status & test accuracy
    metadata_path = os.path.join(config.SAVED_MODELS_DIR, 'model_metadata.json')
    model_trained = os.path.exists(config.MODEL_PATH) and os.path.exists(metadata_path)
    test_accuracy = None
    model_meta = {}
    
    if model_trained:
        try:
            with open(metadata_path, 'r') as f:
                model_meta = json.load(f)
                test_accuracy = round(model_meta.get('test_accuracy', 0.0) * 100, 2)
        except Exception:
            pass

    stats = get_prediction_stats()

    # Active dataset inspection info
    filepath = CURRENT_DATASET.get('filepath')
    dataset_info = {'filename': CURRENT_DATASET.get('filename', 'company_machine_data.csv'), 'is_demo': CURRENT_DATASET.get('is_demo', True), 'total_rows': 0, 'num_features': 0}
    sensor_trends = {}

    if filepath and os.path.exists(filepath):
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            dataset_info['total_rows'] = len(df)
            num_cols = list(df.select_dtypes(include=['number']).columns)
            sensor_candidates = [c for c in num_cols if c.lower() not in ['id', 'sample_index', 'index', 'timestamp']]
            dataset_info['num_features'] = len(sensor_candidates)
            
            for col in sensor_candidates[:4]:
                sensor_trends[col] = df[col].head(100).tolist()
        except Exception:
            pass

    return jsonify({
        'success': True,
        'model_trained': model_trained,
        'test_accuracy': test_accuracy,
        'model_meta': model_meta,
        'latest_prediction': latest_pred,
        'stats': stats,
        'dataset': dataset_info,
        'sensor_trends': sensor_trends
    })

@app.route('/api/export-report', methods=['GET'])
def export_report():
    """Export prediction history as downloadable CSV."""
    logs = get_predictions(limit=1000)
    report_path = os.path.join(config.UPLOADS_DIR, 'equipment_health_report.csv')
    
    import pandas as pd
    if logs:
        clean_logs = []
        for l in logs:
            clean_logs.append({
                'Prediction_ID': l['id'],
                'Timestamp': l['timestamp'],
                'Machine_ID': l.get('machine_id', 'M001'),
                'Dataset_Name': l['dataset_name'],
                'Predicted_Condition': l['predicted_class'],
                'Confidence_Pct': l['confidence'],
                'Health_Score': l.get('health_score', 100.0),
                'Health_Status': l['health_status'],
                'Model_Version': l.get('model_version', 'CNN-LSTM-v1')
            })
        df = pd.DataFrame(clean_logs)
    else:
        df = pd.DataFrame(columns=['Prediction_ID', 'Timestamp', 'Machine_ID', 'Dataset_Name', 'Predicted_Condition', 'Confidence_Pct', 'Health_Score', 'Health_Status', 'Model_Version'])
        
    df.to_csv(report_path, index=False)
    return send_file(report_path, as_attachment=True, download_name='equipment_health_report.csv')

if __name__ == '__main__':
    print("Starting Real Company Machine Data Fault Prediction System on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
