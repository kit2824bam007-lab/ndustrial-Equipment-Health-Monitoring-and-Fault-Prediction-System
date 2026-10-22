import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import config
from data_loader import clean_and_sort_data
from sequence_generator import create_machine_aware_sequences
from model import build_cnn_lstm_model
from evaluate import evaluate_heldout_test
from utils.database import log_training_run

class CustomProgressCallback(tf.keras.callbacks.Callback):
    """Callback to capture per-epoch training progress for Flask live monitoring."""
    def __init__(self, external_callback=None):
        super().__init__()
        self.external_callback = external_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_info = {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss', 0.0)),
            'accuracy': float(logs.get('accuracy', 0.0)),
            'val_loss': float(logs.get('val_loss', 0.0)),
            'val_accuracy': float(logs.get('val_accuracy', 0.0))
        }
        if self.external_callback:
            self.external_callback(epoch_info)

def run_company_training_pipeline(file_path, target_column, feature_columns, sequence_length=30, test_size=0.15, val_size=0.15, epochs=30, batch_size=32, learning_rate=0.001, timestamp_column=None, machine_id_column=None, progress_callback=None):
    """
    Complete end-to-end training pipeline for real company machine datasets:
    1. Pre-training Dataset Validation
    2. Data Cleaning & Chronological Machine Sorting
    3. Machine-Aware 3D Sequence Generation (Zero cross-machine sequence overlapping)
    4. Stratified Train / Validation / Test Splitting (Guarantees class representation in all splits)
    5. StandardScaler fitting STRICTLY on training split (Data Leakage Prevention)
    6. Dynamic 1D CNN + LSTM Keras Model Compilation & Training
    7. Best Model Checkpointing & Early Stopping
    8. Persistence of 6 Artifacts (model.keras, scaler.pkl, encoder.pkl, feature_config.json, history.json, model_metadata.json)
    9. Held-out Test Data Evaluation
    10. SQLite Database Logging
    """
    # 1. Validation & Inspection
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Uploaded CSV dataset file not found at '{file_path}'.")

    try:
        df_raw = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"CSV is unreadable or corrupted: {str(e)}")

    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' missing. Please select a valid target column.")

    missing_features = [f for f in feature_columns if f not in df_raw.columns]
    if missing_features:
        raise ValueError(f"Selected feature column(s) {missing_features} missing from uploaded dataset.")

    if len(feature_columns) == 0:
        raise ValueError("No suitable numeric sensor columns were selected.")

    # 2. Clean and Sort
    df_clean = clean_and_sort_data(
        df=df_raw,
        timestamp_col=timestamp_column,
        machine_id_col=machine_id_column,
        feature_columns=feature_columns
    )

    # Label Encoding
    raw_labels = df_clean[target_column].astype(str).values
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(raw_labels)
    class_names = [str(c) for c in encoder.classes_]

    if len(class_names) < 2:
        raise ValueError(f"Training cannot start: Selected target '{target_column}' contains only 1 class ('{class_names[0]}'). At least two distinct fault classes are required for classification.")

    df_clean['_encoded_target'] = encoded_labels

    # 3. Machine-Aware 3D Sequence Window Generation
    seq_res = create_machine_aware_sequences(
        df=df_clean,
        feature_columns=feature_columns,
        target_column='_encoded_target',
        sequence_length=sequence_length,
        machine_id_column=machine_id_column
    )

    X_seq = seq_res['X']
    y_seq = seq_res['y']
    total_samples = len(X_seq)

    # 4. Stratified Train / Val / Test Split
    # First split out test set
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
    )

    # Calculate relative validation size
    relative_val_size = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42, stratify=y_train_val
    )

    n_train = len(X_train_raw)
    n_val = len(X_val_raw)
    n_test = len(X_test_raw)

    if n_train < 10:
        raise ValueError(f"Dataset contains only {total_samples} generated sequences, resulting in too few training samples ({n_train}). Please upload a larger dataset or reduce sequence length.")

    # 5. Feature Scaling (Data Leakage Prevention)
    num_features = len(feature_columns)
    scaler = StandardScaler()

    # Reshape train to 2D to fit scaler
    X_train_2d = X_train_raw.reshape(-1, num_features)
    scaler.fit(X_train_2d)

    # Transform Train, Val, Test using fitted scaler
    X_train = scaler.transform(X_train_2d).reshape(n_train, sequence_length, num_features)
    X_val = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(n_val, sequence_length, num_features)
    X_test = scaler.transform(X_test_raw.reshape(-1, num_features)).reshape(n_test, sequence_length, num_features)

    # 6. Build Model
    model = build_cnn_lstm_model(
        sequence_length=sequence_length,
        num_features=num_features,
        num_classes=len(class_names),
        learning_rate=learning_rate
    )

    # 7. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.DEFAULT_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    if progress_callback:
        callbacks.append(CustomProgressCallback(progress_callback))

    # 8. Train Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure best model is saved
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)

    # 9. Save All 6 Model Artifacts
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoder, config.ENCODER_PATH)

    feature_config = {
        'target_column': target_column,
        'feature_columns': feature_columns,
        'timestamp_column': timestamp_column,
        'machine_id_column': machine_id_column,
        'sequence_length': sequence_length
    }
    feature_config_path = os.path.join(config.SAVED_MODELS_DIR, 'feature_config.json')
    with open(feature_config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)

    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    history_json_path = os.path.join(config.SAVED_MODELS_DIR, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # 10. Held-out Test Data Evaluation
    eval_results = evaluate_heldout_test(model, X_test, y_test, class_names)

    model_metadata = {
        'model_type': 'CNN-LSTM',
        'dataset_name': os.path.basename(file_path),
        'sequence_length': sequence_length,
        'features': feature_columns,
        'target_column': target_column,
        'timestamp_column': timestamp_column,
        'machine_id_column': machine_id_column,
        'classes': class_names,
        'train_size': n_train,
        'validation_size': n_val,
        'test_size': n_test,
        'test_accuracy': float(eval_results['test_accuracy']),
        'test_precision': float(eval_results['test_precision']),
        'test_recall': float(eval_results['test_recall']),
        'test_f1': float(eval_results['test_f1'])
    }
    metadata_path = os.path.join(config.SAVED_MODELS_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    metrics_payload = {
        'trained': True,
        'dataset_name': os.path.basename(file_path),
        'metadata': model_metadata,
        'test_accuracy': float(eval_results['test_accuracy']),
        'test_precision': float(eval_results['test_precision']),
        'test_recall': float(eval_results['test_recall']),
        'test_f1': float(eval_results['test_f1']),
        'class_names': class_names,
        'confusion_matrix': eval_results['confusion_matrix'],
        'per_class_metrics': eval_results['per_class_metrics'],
        'history': history_dict
    }

    # Save to metrics.json for compatibility
    with open(config.METRICS_PATH, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    # Log experiment run to SQLite DB
    log_training_run(
        dataset_name=os.path.basename(file_path),
        target_column=target_column,
        sequence_length=sequence_length,
        epochs=len(history_dict['loss']),
        metrics=metrics_payload
    )

    return metrics_payload
