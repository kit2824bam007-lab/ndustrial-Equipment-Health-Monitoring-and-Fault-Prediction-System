import os
import json
import config
from utils.database import init_db
from models.train import run_training_pipeline
from models.predict import predict_equipment_condition

def test_full_ml_pipeline():
    print("Initializing SQLite DB...")
    init_db()

    sample_csv = os.path.join(config.SAMPLE_DIR, 'sample_machine_data.csv')
    print(f"Testing training on sample CSV: {sample_csv}")

    target_column = 'fault_label'
    feature_columns = ['vibration_drive_end', 'vibration_fan_end', 'temperature', 'motor_current', 'rotational_speed']
    sequence_length = 100
    epochs = 5

    def progress_cb(info):
        print(f"Epoch {info['epoch']}/{epochs} - loss: {info['loss']:.4f} - acc: {info['accuracy']:.4f} - val_loss: {info['val_loss']:.4f} - val_acc: {info['val_accuracy']:.4f}")

    metrics = run_training_pipeline(
        file_path=sample_csv,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=32,
        progress_callback=progress_cb
    )

    print("\n--- Training Results ---")
    print(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Test Precision: {metrics['test_precision'] * 100:.2f}%")
    print(f"Test Recall: {metrics['test_recall'] * 100:.2f}%")
    print(f"Test F1: {metrics['test_f1'] * 100:.2f}%")
    print("Class Names:", metrics['class_names'])
    print("Confusion Matrix:", metrics['confusion_matrix'])

    print("\n--- Testing Prediction ---")
    pred_res = predict_equipment_condition(sample_csv)
    print(f"Predicted Class: {pred_res['predicted_class']}")
    print(f"Confidence: {pred_res['confidence']}%")
    print(f"Health Status: {pred_res['health_status']}")
    print(f"Alert: {pred_res['alert']['title']}")
    print(f"Probabilities: {pred_res['probabilities']}")
    print("\nPipeline test completed successfully!")

if __name__ == '__main__':
    test_full_ml_pipeline()
