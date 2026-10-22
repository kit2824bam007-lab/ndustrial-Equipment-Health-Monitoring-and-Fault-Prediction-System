import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from data_loader import clean_and_sort_data
from sequence_generator import create_machine_aware_sequences
from utils.helpers import determine_health_status, calculate_health_score, generate_health_alert
from utils.database import log_prediction, log_alert

def predict_machine_health(file_path):
    """
    Execute equipment condition prediction on an unseen company machine CSV dataset.
    Loads saved 6 model artifacts (keras model, scaler.pkl, encoder.pkl, feature_config.json, history.json, metadata.json).
    Performs machine-boundary sequence windowing, softmax probability computation,
    Health Score (0-100) calculation, and SQLite logging.
    """
    # 1. Verify model artifacts exist
    feature_config_path = os.path.join(config.SAVED_MODELS_DIR, 'feature_config.json')
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH) and os.path.exists(config.ENCODER_PATH) and os.path.exists(feature_config_path)):
        raise FileNotFoundError("Trained model artifacts not found. Please train a CNN-LSTM model on a dataset first.")

    # 2. Load artifacts
    model = tf.keras.models.load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoder = joblib.load(config.ENCODER_PATH)

    with open(feature_config_path, 'r') as f:
        feature_config = json.load(f)

    target_col = feature_config.get('target_column')
    feature_cols = feature_config.get('feature_columns', [])
    timestamp_col = feature_config.get('timestamp_column')
    machine_id_col = feature_config.get('machine_id_column')
    seq_len = int(feature_config.get('sequence_length', 30))

    class_names = [str(c) for c in encoder.classes_]

    # 3. Read input CSV
    df_raw = pd.read_csv(file_path)

    missing_cols = [c for c in feature_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Input CSV dataset is missing trained sensor feature column(s): {missing_cols}")

    # Clean and sort input dataset
    df_clean = clean_and_sort_data(
        df=df_raw,
        timestamp_col=timestamp_col,
        machine_id_col=machine_id_col,
        feature_columns=feature_cols
    )

    # 4. Generate Machine-Aware Sequences
    seq_res = create_machine_aware_sequences(
        df=df_clean,
        feature_columns=feature_cols,
        target_column=None,
        sequence_length=seq_len,
        machine_id_column=machine_id_col
    )

    X_seq_raw = seq_res['X']
    n_samples = len(X_seq_raw)
    num_features = len(feature_cols)

    # Scale using saved training scaler
    X_seq_2d = X_seq_raw.reshape(-1, num_features)
    X_seq_scaled = scaler.transform(X_seq_2d).reshape(n_samples, seq_len, num_features)

    # 5. Model Inference
    all_probs = model.predict(X_seq_scaled, verbose=0)
    
    # Calculate mean prediction probabilities across time sequences
    mean_probs = np.mean(all_probs, axis=0)
    pred_class_idx = int(np.argmax(mean_probs))
    confidence_val = float(mean_probs[pred_class_idx]) * 100.0

    predicted_class = str(encoder.inverse_transform([pred_class_idx])[0])

    # 6. Calculate Health Score (0-100) & Health Status
    health_score = calculate_health_score(predicted_class, confidence_val)
    health_status = determine_health_status(predicted_class, health_score)

    alert_obj = generate_health_alert(health_status, predicted_class, health_score)

    # Map probability dictionary
    prob_dict = {}
    for idx, c_name in enumerate(class_names):
        prob_dict[c_name] = round(float(mean_probs[idx]) * 100, 2)

    # Extract latest timestamp and machine ID if available
    latest_timestamp = str(df_clean[timestamp_col].iloc[-1]) if timestamp_col and timestamp_col in df_clean.columns else "N/A"
    machine_id_val = str(df_clean[machine_id_col].iloc[-1]) if machine_id_col and machine_id_col in df_clean.columns else "M001"

    dataset_name = os.path.basename(file_path)

    # Sample sensor trend data for interactive charts
    sensor_trends = {}
    for col in feature_cols[:4]:
        sensor_trends[col] = df_clean[col].head(100).tolist()

    # 7. Log Prediction to SQLite Database
    pred_id = log_prediction(
        dataset_name=dataset_name,
        predicted_class=predicted_class,
        confidence=confidence_val,
        health_status=health_status,
        probabilities=prob_dict,
        sensor_values=sensor_trends,
        machine_id=machine_id_val,
        timestamp_str=latest_timestamp,
        health_score=health_score
    )

    # Log Alert if WARNING or CRITICAL
    if health_status in ['WARNING', 'CRITICAL']:
        log_alert(
            machine_id=machine_id_val,
            predicted_class=predicted_class,
            confidence=confidence_val,
            health_status=health_status,
            health_score=health_score,
            message=alert_obj['message']
        )

    return {
        'prediction_id': pred_id,
        'dataset_name': dataset_name,
        'machine_id': machine_id_val,
        'timestamp': latest_timestamp,
        'predicted_class': predicted_class,
        'confidence': round(confidence_val, 2),
        'health_score': health_score,
        'health_status': health_status,
        'alert': alert_obj,
        'probabilities': prob_dict,
        'class_names': class_names,
        'feature_columns': feature_cols,
        'sensor_trends': sensor_trends,
        'total_windows_analyzed': n_samples
    }

# Alias for backward compatibility
predict_equipment_condition = predict_machine_health
