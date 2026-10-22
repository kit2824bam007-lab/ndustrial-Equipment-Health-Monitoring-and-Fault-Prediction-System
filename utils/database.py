import sqlite3
import json
import os
from datetime import datetime
import config

def get_db_connection():
    """Connect to SQLite database and return connection object."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database tables and auto-migrate missing columns if necessary."""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Predictions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dataset_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            health_status TEXT NOT NULL,
            probabilities_json TEXT,
            sensor_values_json TEXT,
            model_version TEXT DEFAULT 'CNN-LSTM-v1'
        )
    ''')

    # Migration for new company machine prediction columns
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
    if 'machine_id' not in existing_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN machine_id TEXT DEFAULT 'M001'")
    if 'timestamp_str' not in existing_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN timestamp_str TEXT")
    if 'health_score' not in existing_cols:
        cursor.execute("ALTER TABLE predictions ADD COLUMN health_score REAL DEFAULT 100.0")

    # Alerts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine_id TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            health_status TEXT NOT NULL,
            health_score REAL NOT NULL,
            message TEXT NOT NULL,
            recommendation TEXT DEFAULT ''
        )
    ''')

    existing_alert_cols = [row[1] for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()]
    if 'recommendation' not in existing_alert_cols:
        cursor.execute("ALTER TABLE alerts ADD COLUMN recommendation TEXT DEFAULT ''")

    # Training Runs Table (Multi-Experiment History)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dataset_name TEXT NOT NULL,
            target_column TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            epochs INTEGER NOT NULL,
            test_accuracy REAL,
            test_precision REAL,
            test_recall REAL,
            test_f1 REAL,
            metrics_json TEXT
        )
    ''')

    conn.commit()
    conn.close()

def log_prediction(dataset_name, predicted_class, confidence, health_status, probabilities=None, sensor_values=None, machine_id="M001", timestamp_str=None, health_score=100.0, model_version="CNN-LSTM-v1"):
    """Log a prediction event to SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    prob_json = json.dumps(probabilities) if probabilities is not None else "{}"
    sensor_json = json.dumps(sensor_values) if sensor_values is not None else "{}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO predictions (timestamp, dataset_name, machine_id, timestamp_str, predicted_class, confidence, health_status, health_score, probabilities_json, sensor_values_json, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now_str, dataset_name, machine_id, timestamp_str, predicted_class, float(confidence), health_status, float(health_score), prob_json, sensor_json, model_version))

    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id

def log_alert(machine_id, predicted_class, confidence, health_status, health_score, message, recommendation=""):
    """Log warning or critical alert event to SQLite DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO alerts (timestamp, machine_id, predicted_class, confidence, health_status, health_score, message, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now_str, machine_id, predicted_class, float(confidence), health_status, float(health_score), message, recommendation))

    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()
    return alert_id

def clear_alerts():
    """Clear all recorded asset alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()

def log_training_run(dataset_name, target_column, sequence_length, epochs, metrics):
    """Log a completed model training session."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_json_str = json.dumps(metrics)

    test_acc = metrics.get('test_accuracy', 0.0)
    test_prec = metrics.get('test_precision', 0.0)
    test_rec = metrics.get('test_recall', 0.0)
    test_f1 = metrics.get('test_f1', 0.0)

    cursor.execute('''
        INSERT INTO training_runs (timestamp, dataset_name, target_column, sequence_length, epochs, test_accuracy, test_precision, test_recall, test_f1, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now_str, dataset_name, target_column, sequence_length, epochs, test_acc, test_prec, test_rec, test_f1, metrics_json_str))

    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id

def get_predictions(limit=100, fault_class=None, health_status=None, start_date=None, end_date=None):
    """Retrieve prediction history filtered by class, health status, or date range."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if fault_class:
        query += " AND predicted_class = ?"
        params.append(fault_class)
    if health_status:
        query += " AND health_status = ?"
        params.append(health_status)
    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date} 00:00:00")
    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date} 23:59:59")

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        item = dict(row)
        item['probabilities'] = json.loads(item['probabilities_json']) if item.get('probabilities_json') else {}
        item['sensor_values'] = json.loads(item['sensor_values_json']) if item.get('sensor_values_json') else {}
        results.append(item)
    return results

def get_alert_history(limit=100):
    """Retrieve recorded alert logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_latest_prediction():
    """Retrieve the single most recent prediction."""
    preds = get_predictions(limit=1)
    return preds[0] if preds else None

def get_latest_training_run():
    """Retrieve details of most recent training run."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None
    res = dict(row)
    if res.get('metrics_json'):
        res['metrics'] = json.loads(res['metrics_json'])
    return res

def get_prediction_stats():
    """Calculate summary statistics from predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT health_status, COUNT(*) as count FROM predictions GROUP BY health_status")
    health_counts = {row['health_status']: row['count'] for row in cursor.fetchall()}

    cursor.execute("SELECT predicted_class, COUNT(*) as count FROM predictions GROUP BY predicted_class")
    class_counts = {row['predicted_class']: row['count'] for row in cursor.fetchall()}

    conn.close()
    return {
        'total_predictions': total,
        'health_counts': health_counts,
        'class_counts': class_counts
    }
