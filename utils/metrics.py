import numpy as np

def format_evaluation_summary(metrics_dict):
    """
    Format precision, recall, f1-score, accuracy into rounded display dictionary.
    """
    if not metrics_dict:
        return None
    
    return {
        'test_accuracy': round(metrics_dict.get('test_accuracy', 0.0) * 100, 2),
        'test_precision': round(metrics_dict.get('test_precision', 0.0) * 100, 2),
        'test_recall': round(metrics_dict.get('test_recall', 0.0) * 100, 2),
        'test_f1': round(metrics_dict.get('test_f1', 0.0) * 100, 2),
        'confusion_matrix': metrics_dict.get('confusion_matrix', []),
        'class_names': metrics_dict.get('class_names', []),
        'per_class_metrics': metrics_dict.get('per_class_metrics', {})
    }

def sanitize_confusion_matrix(cm):
    """Convert numpy confusion matrix to nested Python list for JSON response."""
    if isinstance(cm, np.ndarray):
        return cm.tolist()
    return cm
