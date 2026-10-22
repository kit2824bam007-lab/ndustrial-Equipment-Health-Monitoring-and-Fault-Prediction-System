import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate trained CNN-LSTM model on unseen test set.
    Computes accuracy, weighted precision, recall, F1-score,
    per-class Breakdown, and confusion matrix array.
    """
    # Predict probabilities
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    per_class_metrics = {}
    for cls in class_names:
        if cls in report:
            per_class_metrics[cls] = {
                'precision': float(report[cls]['precision']),
                'recall': float(report[cls]['recall']),
                'f1-score': float(report[cls]['f1-score']),
                'support': int(report[cls]['support'])
            }

    return {
        'test_accuracy': acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'confusion_matrix': cm,
        'class_names': class_names,
        'per_class_metrics': per_class_metrics
    }
