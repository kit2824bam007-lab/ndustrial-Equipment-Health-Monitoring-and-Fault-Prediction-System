# 🏭 Industrial Equipment Health Monitoring and Fault Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-green?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A complete end-to-end predictive maintenance platform powered by a hybrid 1D CNN + LSTM deep learning architecture. Detects and classifies industrial equipment faults from time-series sensor data — with a professional, real-time web dashboard.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture-cnn--lstm)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Dataset Support](#-dataset-support)
- [Installation & Setup](#-installation--setup)
- [Web Interface](#-web-interface-pages)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Academic Disclaimers](#-academic-disclaimers)

---

## 🔍 Project Overview

This system implements a **fully software-based predictive maintenance pipeline** that processes industrial machine sensor data (vibration, temperature, motor current, speed, pressure) and classifies equipment health in real time using deep learning.

**End-to-end Pipeline:**

```
Raw Sensor CSV  →  Preprocessing  →  1D CNN  →  LSTM  →  Fault Class  →  Health Status  →  Dashboard
```

| Health Status | Description |
|:---:|:---|
| 🟢 **NORMAL** | Equipment operating within baseline parameters |
| 🟡 **WARNING** | Early-stage fault indicators detected (e.g., inner race, ball fault) |
| 🔴 **CRITICAL** | Severe fault condition — immediate maintenance required |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAULT PREDICTION SYSTEM                      │
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────────┐  │
│  │  CSV     │───▶│Preprocessing│───▶│   CNN-LSTM Model       │  │
│  │  Upload  │    │  & Scaling  │    │ (TensorFlow/Keras)     │  │
│  └──────────┘    └─────────────┘    └──────────┬────────────┘  │
│                                                 │               │
│  ┌──────────────────────────────────────────────▼────────────┐  │
│  │              Flask REST API Backend                        │  │
│  │  /api/upload  /api/train  /api/predict  /api/dashboard    │  │
│  └──────────────────────────────────────────────┬────────────┘  │
│                                                 │               │
│  ┌──────────────┐    ┌────────────────┐         │               │
│  │   SQLite DB  │◀───│  Prediction    │◀────────┘               │
│  │  (Logs &     │    │  Logger &      │                         │
│  │   Alerts)    │    │  Health Engine │                         │
│  └──────────────┘    └────────────────┘                         │
│                                                 │               │
│  ┌──────────────────────────────────────────────▼────────────┐  │
│  │           Web Dashboard (HTML5 + Chart.js)                 │  │
│  │   Dashboard | Dataset | Training | Prediction | Reports   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture: CNN + LSTM

The model combines **Convolutional Neural Networks** for local feature extraction and **Long Short-Term Memory** networks for temporal dependency learning:

```
Input Shape: (sequence_length, num_features)
        │
        ▼
┌───────────────────────┐
│  Conv1D (32 filters)  │  ← Extracts localized harmonic patterns
│  kernel_size=3        │
│  BatchNormalization   │
│  ReLU Activation      │
│  MaxPooling1D (2)     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Conv1D (64 filters)  │  ← Deep feature abstraction
│  kernel_size=3        │
│  BatchNormalization   │
│  ReLU Activation      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   LSTM (64 units)     │  ← Temporal degradation dynamics
│   return_sequences=F  │
│   Dropout (0.3)       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Dense (64, ReLU)    │  ← Classification head
│   Dense (N, Softmax)  │  ← N = number of fault classes
└───────────────────────┘

Optimizer: Adam | Loss: Sparse Categorical Crossentropy
```

### Why CNN + LSTM?
- **1D CNN**: Efficiently captures high-frequency vibration harmonics and impact signatures in sliding time windows
- **LSTM**: Learns temporal patterns and gradual machine condition degradation over time
- **Combined**: Achieves higher fault classification accuracy than either architecture alone

---

## 🛠️ Technology Stack

| Category | Technologies |
|:---|:---|
| **Language** | Python 3.12 |
| **Backend Framework** | Flask 3.0+, Flask-CORS |
| **Deep Learning** | TensorFlow 2.16+, Keras (Conv1D, BatchNorm, LSTM, Dense) |
| **Data Science** | NumPy, Pandas, SciPy, Scikit-learn, Joblib |
| **Database** | SQLite3 (prediction logs, alert history) |
| **Frontend** | HTML5, Vanilla CSS (Dark Industrial Slate Theme), JavaScript ES6+ |
| **Visualization** | Chart.js (real-time sensor trends, confusion matrix, loss curves) |
| **Model Persistence** | Keras `.keras` format, Joblib `.pkl` (scaler, label encoder) |

---

## 📁 Project Structure

```
FAULT PREDICTION SYSTEM/
│
├── 📄 app.py                          # Flask application — all routes & API endpoints
├── 📄 config.py                       # Centralized configuration (paths, ML params, health mappings)
├── 📄 model.py                        # Model utility wrapper
├── 📄 data_loader.py                  # CSV inspection & company data loader
├── 📄 evaluate.py                     # Standalone model evaluation script
├── 📄 sequence_generator.py           # Sliding window sequence generator
│
├── 📄 create_sample_data.py           # Generates sample_machine_data.csv for testing
├── 📄 create_company_sample_data.py   # Generates company_machine_data.csv (realistic demo)
├── 📄 test_pipeline.py                # End-to-end pipeline test script
├── 📄 test_company_pipeline.py        # Company dataset pipeline test
│
├── 📄 requirements.txt                # Python dependencies
│
├── 📂 models/                         # Core ML pipeline modules
│   ├── __init__.py
│   ├── cnn_lstm.py                    # CNN-LSTM model architecture builder
│   ├── preprocessing.py               # Data cleaning, scaling, sequence creation
│   ├── train.py                       # Training pipeline (fit, evaluate, save)
│   ├── predict.py                     # Inference pipeline (load model, run predictions)
│   └── evaluation.py                  # Metrics computation (accuracy, F1, confusion matrix)
│
├── 📂 utils/                          # Shared utilities
│   ├── __init__.py
│   ├── database.py                    # SQLite schema, CRUD for predictions & alerts
│   ├── helpers.py                     # File validation, formatting helpers
│   └── metrics.py                     # Health score computation
│
├── 📂 static/                         # Frontend static assets
│   ├── css/
│   │   └── style.css                  # Dark industrial theme stylesheet
│   └── js/
│       ├── dashboard.js               # Dashboard charts & live KPI updates
│       ├── dataset.js                 # Dataset upload & configuration UI
│       ├── training.js                # Training form, live logs, metric charts
│       ├── prediction.js              # Prediction form & confidence display
│       ├── history.js                 # Filterable prediction history table
│       └── reports.js                 # Report generation & CSV export
│
├── 📂 templates/                      # Jinja2 HTML templates
│   ├── base.html                      # Base layout with sidebar navigation
│   ├── dashboard.html                 # Main monitoring dashboard
│   ├── dataset.html                   # Dataset upload & preview page
│   ├── training.html                  # Model training & evaluation page
│   ├── prediction.html                # Real-time fault prediction page
│   ├── history.html                   # Prediction history logs page
│   ├── reports.html                   # Executive reports & export page
│   └── about.html                     # System methodology & architecture info
│
├── 📂 data/                           # Runtime data directories (auto-created)
│   ├── sample/                        # Built-in demo datasets
│   ├── uploads/                       # User-uploaded CSV files
│   └── processed/                     # Intermediate preprocessed data
│
├── 📂 saved_models/                   # Trained model artifacts (auto-created)
│   ├── cnn_lstm_model.keras           # Trained Keras model
│   ├── scaler.pkl                     # Fitted StandardScaler
│   ├── label_encoder.pkl              # Fitted LabelEncoder
│   ├── metrics.json                   # Training/evaluation metrics
│   └── model_metadata.json            # Model version & configuration metadata
│
└── 📂 database/                       # SQLite database (auto-created)
    └── database.db                    # Prediction logs & alert history
```

---

## 📊 Dataset Support

### Supported Input Format
- **File Format**: Standard CSV (`.csv`)
- **Max Upload Size**: 100 MB
- **Required**: At minimum one numeric sensor column and one categorical fault label column

### Built-in Demo Datasets
| Dataset | Description |
|:---|:---|
| `company_machine_data.csv` | Realistic multi-machine industrial dataset (primary demo) |
| `sample_machine_data.csv` | Simplified baseline dataset for quick testing |

### Recommended Real-World Datasets
| Dataset | Source | Use Case |
|:---|:---|:---|
| **CWRU Bearing Dataset** | [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter) | Vibration fault classification (Normal, Inner Race, Outer Race, Ball Fault) |
| **NASA C-MAPSS** | [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division) | Turbofan engine degradation (RUL regression — separate mode) |
| **Custom Industrial CSV** | Your own equipment sensors | Any multi-sensor time-series with fault labels |

### CSV Column Requirements
```
Recommended columns (auto-detected):
  - Numeric sensor readings: vibration_x, temperature, current, speed, pressure, ...
  - Fault label column:      fault_type, condition, label, status (categorical)
  - Optional: timestamp, machine_id (for multi-machine datasets)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python **3.10 – 3.12** (TensorFlow 2.16+ requires Python ≤ 3.12)
- pip package manager
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/kit2824bam007-lab/ndustrial-Equipment-Health-Monitoring-and-Fault-Prediction-System.git
cd ndustrial-Equipment-Health-Monitoring-and-Fault-Prediction-System
```

### Step 2 — Create Virtual Environment *(Recommended)*
```bash
# Create venv
python -m venv venv

# Activate — Windows PowerShell
.\venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate Demo Data *(Optional)*
```bash
# Generate realistic company demo dataset
python create_company_sample_data.py

# Or the minimal sample dataset
python create_sample_data.py
```

### Step 5 — Run the Application
```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🌐 Web Interface Pages

| Page | Route | Description |
|:---|:---|:---|
| **Dashboard** | `/dashboard` | Live KPI cards (Health Status, Predicted Fault, Confidence %, Model Accuracy), sensor trend charts, alert feed |
| **Dataset** | `/dataset` | Drag-and-drop CSV upload, dataset statistics preview, column mapping selector, sequence length config |
| **Training** | `/training` | Configure epochs/batch size/learning rate, live terminal log, Loss & Accuracy curves, Confusion Matrix |
| **Prediction** | `/prediction` | Upload CSV for inference, fault class probability chart, health status badge, sensor signal viewer |
| **History** | `/history` | SQLite prediction log with date range, health status, and fault class filters |
| **Reports** | `/reports` | Executive summary report with downloadable CSV export |
| **About** | `/about` | System methodology, architecture diagram, and technology breakdown |

---

## 📡 API Reference

All API endpoints return JSON responses.

| Method | Endpoint | Description |
|:---:|:---|:---|
| `POST` | `/api/upload-dataset` | Upload a company CSV dataset |
| `GET` | `/api/inspect-current-dataset` | Inspect the currently active dataset |
| `POST` | `/api/use-company-sample` | Switch to the built-in demo dataset |
| `POST` | `/api/prepare-dataset` | Preview sequence windows, class distribution & split sizes |
| `POST` | `/api/train` | Trigger CNN-LSTM training on the active dataset |
| `POST` | `/api/predict` | Run fault prediction inference |
| `GET` | `/api/training-history` | Retrieve saved model metrics and training curves |
| `GET` | `/api/predictions` | Fetch filterable prediction logs from SQLite |
| `GET` | `/api/alerts` | Fetch warning/critical alert history |
| `GET` | `/api/dashboard-summary` | Aggregated KPI stats for the dashboard |
| `GET` | `/api/export-report` | Download prediction history as CSV |

---

## 🔧 Configuration

Key settings in `config.py`:

```python
# Default ML Hyperparameters
DEFAULT_SEQUENCE_LENGTH = 100   # Sliding window size (timesteps)
DEFAULT_EPOCHS          = 30    # Training epochs
DEFAULT_BATCH_SIZE      = 32    # Mini-batch size
DEFAULT_PATIENCE        = 5     # Early stopping patience

# Health Status Mapping (fully customizable)
HEALTH_STATUS_MAPPING = {
    'normal':       'NORMAL',
    'inner race':   'WARNING',
    'ball fault':   'WARNING',
    'outer race':   'CRITICAL',
    'severe fault': 'CRITICAL',
    'failure':      'CRITICAL',
}

# Upload Limits
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB
```

---

## 📜 Academic Disclaimers

- **Software-Only Architecture**: This project requires **no physical sensors, Arduino, Raspberry Pi, or PLC hardware**. It operates entirely on uploaded or replayed CSV datasets.
- **Non-Fabricated Metrics**: All evaluation scores, accuracy curves, confusion matrices, and confidence percentages are computed directly from model training and inference — never hardcoded or simulated.
- **Data Leakage Prevention**: The `StandardScaler` is fitted exclusively on the training split and applied to validation/test splits, following standard ML best practices.
- **CWRU Bearing Dataset**: If using CWRU data, please cite the original data center: *Case Western Reserve University Bearing Data Center, Seeded Fault Test Data.*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

<div align="center">

**Built with ❤️ for Industrial Predictive Maintenance**

*Python · TensorFlow · Flask · SQLite · Chart.js*

</div>
