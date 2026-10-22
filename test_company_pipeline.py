import os
import json
import config
from utils.database import init_db
from data_loader import inspect_company_csv
from models.preprocessing import preview_dataset_preparation
from models.train import run_company_training_pipeline
from models.predict import predict_machine_health

def test_company_pipeline():
    print("=== Testing Real Company Machine Data Pipeline ===")
    init_db()

    csv_path = os.path.join(config.SAMPLE_DIR, 'company_machine_data.csv')
    print(f"\n1. Inspecting dataset: {csv_path}")
    inspection = inspect_company_csv(csv_path)
    print(f"Total Rows: {inspection['total_rows']}, Cols: {inspection['total_columns']}")
    print(f"Machine ID candidates: {inspection['machine_id_candidates']}")
    print(f"Datetime candidates: {inspection['datetime_candidates']}")
    print(f"Numeric signals: {inspection['numeric_columns']}")
    print(f"Target candidates: {inspection['target_candidates']}")

    target_col = 'Fault_Status'
    feature_cols = ['Temperature', 'Vibration', 'Current', 'RPM', 'Pressure', 'Load', 'Operating_Hours']
    machine_id_col = 'Machine_ID'
    timestamp_col = 'Timestamp'
    seq_len = 30
    epochs = 5

    print("\n2. Previewing Dataset Preparation & Sequence Generation...")
    prep = preview_dataset_preparation(
        file_path=csv_path,
        target_column=target_col,
        feature_columns=feature_cols,
        sequence_length=seq_len,
        test_size=0.15,
        val_size=0.15,
        machine_id_column=machine_id_col,
        timestamp_column=timestamp_col
    )
    print(f"Clean Rows: {prep['clean_rows']}, Removed Duplicates: {prep['removed_duplicates']}")
    print(f"Total 3D Sequences: {prep['total_sequences']} across {prep['valid_machines_count']} valid machines")
    print(f"Split Counts -> Train: {prep['train_samples']}, Val: {prep['val_samples']}, Test: {prep['test_samples']}")
    print(f"Classes ({prep['num_classes']}): {prep['class_names']}")
    print(f"Distribution: {prep['class_distribution']}")

    print("\n3. Running Training Pipeline...")
    def cb(info):
        print(f"Epoch {info['epoch']}/{epochs} - loss: {info['loss']:.4f} - acc: {info['accuracy']:.4f} - val_loss: {info['val_loss']:.4f} - val_acc: {info['val_accuracy']:.4f}")

    metrics = run_company_training_pipeline(
        file_path=csv_path,
        target_column=target_col,
        feature_columns=feature_cols,
        sequence_length=seq_len,
        test_size=0.15,
        val_size=0.15,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.001,
        timestamp_column=timestamp_col,
        machine_id_column=machine_id_col,
        progress_callback=cb
    )

    print("\n--- Training & Evaluation Results ---")
    print(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Test Precision: {metrics['test_precision'] * 100:.2f}%")
    print(f"Test Recall: {metrics['test_recall'] * 100:.2f}%")
    print(f"Test F1: {metrics['test_f1'] * 100:.2f}%")
    print("Class Names:", metrics['class_names'])
    print("Confusion Matrix:", metrics['confusion_matrix'])

    print("\n4. Verifying All 6 Saved Model Artifacts...")
    expected_artifacts = [
        config.MODEL_PATH,
        config.SCALER_PATH,
        config.ENCODER_PATH,
        os.path.join(config.SAVED_MODELS_DIR, 'feature_config.json'),
        os.path.join(config.SAVED_MODELS_DIR, 'training_history.json'),
        os.path.join(config.SAVED_MODELS_DIR, 'model_metadata.json')
    ]
    for path in expected_artifacts:
        exists = os.path.exists(path)
        print(f"Artifact [{os.path.basename(path)}]: {'EXISTS' if exists else 'MISSING'}")

    print("\n5. Testing Machine Health Prediction Inference...")
    pred_res = predict_machine_health(csv_path)
    print(f"Machine ID: {pred_res['machine_id']}")
    print(f"Timestamp: {pred_res['timestamp']}")
    print(f"Predicted Class: {pred_res['predicted_class']}")
    print(f"Confidence: {pred_res['confidence']}%")
    print(f"Health Score: {pred_res['health_score']}/100")
    print(f"Health Status: {pred_res['health_status']}")
    print(f"Alert Title: {pred_res['alert']['title']}")
    print(f"Probabilities: {pred_res['probabilities']}")

    print("\n=== Company Pipeline Test Succeeded ===")

if __name__ == '__main__':
    test_company_pipeline()
