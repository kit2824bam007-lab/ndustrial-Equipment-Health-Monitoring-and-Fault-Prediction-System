import os

# Base Directory of the Project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SAMPLE_DIR = os.path.join(DATA_DIR, 'sample')

SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'cnn_lstm_model.keras')
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, 'label_encoder.pkl')
METRICS_PATH = os.path.join(SAVED_MODELS_DIR, 'metrics.json')

DATABASE_DIR = os.path.join(BASE_DIR, 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'database.db')

# Ensure directories exist
for directory in [UPLOADS_DIR, PROCESSED_DIR, SAMPLE_DIR, SAVED_MODELS_DIR, DATABASE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default ML Parameters
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_PATIENCE = 5

# Health Status Mappings (Configurable)
# Maps fault class keywords or full class names to Health Status
HEALTH_STATUS_MAPPING = {
    'normal': 'NORMAL',
    'healthy': 'NORMAL',
    'baseline': 'NORMAL',
    'inner race': 'WARNING',
    'ball fault': 'WARNING',
    'minor fault': 'WARNING',
    'outer race': 'CRITICAL',
    'severe fault': 'CRITICAL',
    'failure': 'CRITICAL',
    'broken': 'CRITICAL'
}

# Upload Security Configuration
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload
