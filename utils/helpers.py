import os
import config

def allowed_file(filename):
    """Check if uploaded file has a permitted extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def calculate_health_score(predicted_class, confidence_val):
    """
    Calculate software Health Score on a 0 - 100 scale:
    - 100 = Pristine healthy condition
    - 75 - 90 = Normal baseline condition
    - 40 - 74 = Warning / Minor fault degradation
    - 0 - 39 = Critical fault / Severe failure
    """
    cls_lower = str(predicted_class).strip().lower()
    conf_ratio = float(confidence_val) / 100.0

    if any(k in cls_lower for k in ['normal', 'healthy', 'baseline', '0']):
        score = 85.0 + (15.0 * conf_ratio)
    elif any(k in cls_lower for k in ['warning', 'minor', 'degrad', '1']):
        score = 75.0 - (30.0 * conf_ratio)
    elif any(k in cls_lower for k in ['critical', 'fault', 'severe', 'failure', 'broken', 'outer', 'inner', 'ball', 'overheat']):
        score = max(5.0, 45.0 - (35.0 * conf_ratio))
    else:
        score = 100.0 - (conf_ratio * 50.0)

    return round(float(score), 1)

def determine_health_status(predicted_class, health_score=None):
    """
    Map predicted fault class string and health score to software health status:
    Returns 'NORMAL', 'WARNING', 'CRITICAL', or fallback 'Condition Detected'.
    """
    cls_lower = str(predicted_class).strip().lower()

    if health_score is not None:
        if health_score >= 80.0:
            return 'NORMAL'
        elif health_score >= 45.0:
            return 'WARNING'
        else:
            return 'CRITICAL'

    for keyword, status in config.HEALTH_STATUS_MAPPING.items():
        if keyword in cls_lower:
            return status

    if any(k in cls_lower for k in ['normal', 'healthy', 'baseline', '0']):
        return 'NORMAL'
    elif any(k in cls_lower for k in ['critical', 'fault', 'severe', 'failure', 'broken']):
        return 'CRITICAL'
    elif any(k in cls_lower for k in ['warning', 'minor', 'degrad']):
        return 'WARNING'
    else:
        return 'Condition Detected'

def generate_maintenance_recommendation(predicted_class, health_status, health_score=100.0):
    """Generate actionable enterprise maintenance recommendation based on predicted fault condition."""
    cls_lower = str(predicted_class).strip().lower()

    if health_status == 'NORMAL':
        return "Equipment operating within baseline specifications. Continue routine preventive maintenance schedule."
    
    if 'bearing' in cls_lower or 'race' in cls_lower or 'ball' in cls_lower:
        return "Perform high-frequency vibration FFT analysis. Schedule bearing relubrication or replacement within 48 hours."
    elif 'overheat' in cls_lower or 'thermal' in cls_lower or 'temp' in cls_lower:
        return "Inspect cooling fan airflow, thermal heatsink, and motor current draw immediately to prevent thermal breakdown."
    elif 'motor' in cls_lower or 'current' in cls_lower or 'stator' in cls_lower:
        return "Measure 3-phase winding resistance and check inverter drive output parameters for electrical unbalance."
    elif 'vibration' in cls_lower or 'unbalance' in cls_lower:
        return "Check shaft alignment, rotor balance, and foundation bolt torque specifications."
    else:
        return f"Inspect asset mechanical assemblies and review sensor trend telemetry for '{predicted_class}'."

def generate_health_alert(health_status, predicted_class, health_score=None):
    """Generate alert message object with actionable recommendation."""
    score_str = f" (Health Score: {health_score}/100)" if health_score is not None else ""
    recommendation = generate_maintenance_recommendation(predicted_class, health_status, health_score or 100.0)

    if health_status == 'CRITICAL':
        return {
            'level': 'critical',
            'title': 'CRITICAL ALERT - Severe Fault Detected',
            'message': f"Critical machine fault condition predicted: '{predicted_class}'{score_str}.",
            'recommendation': recommendation
        }
    elif health_status == 'WARNING':
        return {
            'level': 'warning',
            'title': 'WARNING - Asset Degradation Detected',
            'message': f"Abnormal equipment condition detected: '{predicted_class}'{score_str}.",
            'recommendation': recommendation
        }
    elif health_status == 'NORMAL':
        return {
            'level': 'normal',
            'title': 'SYSTEM HEALTHY - Normal Operation',
            'message': f"Equipment operating normally{score_str}. Condition: '{predicted_class}'.",
            'recommendation': recommendation
        }
    else:
        return {
            'level': 'normal',
            'title': 'CONDITION EVALUATED',
            'message': f"Machine condition evaluated as '{predicted_class}'{score_str}.",
            'recommendation': recommendation
        }

def format_file_size(size_in_bytes):
    """Format bytes into readable string KB/MB."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
