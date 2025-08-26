# 🏥 Medical Article Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)
![Multi-Label](https://img.shields.io/badge/Multi--Label-Classification-orange.svg)
![F1 Score](https://img.shields.io/badge/F1--Score-0.8933-brightgreen.svg)

**Advanced Multi-Label Classification for Medical Literature**

[![Model Performance](https://img.shields.io/badge/Weighted%20F1-0.8933-brightgreen)](https://github.com/your-repohttps://github.com/JuanCSanchezH/AI-Biomedical-Classification-Challenge)
[![Hamming Loss](https://img.shields.io/badge/Hamming%20Loss-0.0677-green)](https://github.com/JuanCSanchezH/AI-Biomedical-Classification-Challenge)
[![Training Time](https://img.shields.io/badge/Training%20Time-5.23s-blue)](https://github.com/JuanCSanchezH/AI-Biomedical-Classification-Challenge)

</div>

---

## 🎯 Project Overview

This project implements a state-of-the-art **Binary Relevance + XGBoost** system for classifying medical articles into multiple domains. The system achieves excellent performance with a **Weighted F1 Score of 0.8933** and can classify articles into:

- 🫀 **Cardiovascular**
- 🧠 **Neurological** 
- 🫁 **Hepatorenal**
- 🦠 **Oncological**

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd AI-Biomedical-Classification-Challenge

# Install dependencies
pipenv install

# Activate environment
pipenv shell
```

### Usage Examples

#### 🎯 Single Article Classification
```python
from src.test_model import ModelTester

tester = ModelTester()
result = tester.predict_single(
    title="Cardiac arrhythmia detection using machine learning",
    abstract="This study presents a novel approach for detecting cardiac arrhythmias..."
)

print(f"Predicted labels: {result['predicted_labels']}")
# Output: cardiovascular
```

#### 📚 Batch Classification
```python
titles = ["Title 1", "Title 2", "Title 3"]
abstracts = ["Abstract 1", "Abstract 2", "Abstract 3"]

results = tester.predict_multiple(titles, abstracts)
for result in results:
    print(f"Labels: {result['predicted_labels']}")
```

## 📊 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Weighted F1 Score** | 0.8933 | 🟢 Excellent |
| **Hamming Loss** | 0.0677 | 🟢 Excellent |
| **Subset Accuracy** | 0.7602 | 🟡 Good |
| **Training Time** | 5.23s | 🟢 Fast |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  Preprocessing  │───▶│ Feature Engine  │
│  (Title + Abs)  │    │   (TF-IDF)      │    │   (5000 feat)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Multi-Label     │◀───│ BR + XGBoost    │◀───│   Model         │
│ Output          │    │ Classifier      │    │   Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
AI-Biomedical-Classification-Challenge/
├── 🗂️ input/                    # Dataset
│   └── challenge_data.csv
├── 🤖 models/                   # Trained models
│   ├── BR_xgboost_model.pkl
│   └── feature_pipeline.pkl
├── 📊 output/                   # Results & reports
│   ├── comprehensive_report.png
│   ├── COMPREHENSIVE_REPORT.md
│   ├── model_comparison_table.csv
│   ├── multiple_predictions.csv
│   └── test_predictions.csv
├── 💻 src/                      # Source code
│   ├── data/loader.py
│   ├── features/vectorizer.py
│   ├── models/br_xgboost_model.py
│   ├── train_model.py
│   └── test_model.py
├── 🧪 tests/                    # Unit tests
│   └── test_br_xgboost_model.py
└── 🎨 v0_visualization/         # V0 Dashboard & Demo Package
    ├── 📊 dashboard_data.json          # Comprehensive project insights
    ├── 📈 chart_data.json              # Chart-ready data for V0
    ├── 🤖 real_time_classifier.py      # Full-featured classification API
    ├── 🎮 demo_classifier.py           # Standalone demo (no dependencies)
    ├── 🔌 api_endpoints.json           # API endpoint definitions
    ├── 📖 README.md                    # V0 integration guide
    ├── 📋 SUMMARY.md                   # Package summary
    └── 📁 FILE_STRUCTURE.md            # File structure overview
```

## 🔧 Commands

### Training
```bash
pipenv run python src/train_model.py
```

### Testing
```bash
pipenv run python src/test_model.py
```

### Unit Tests
```bash
pipenv run pytest tests/ -v
```

### V0 Demo
```bash
cd v0_visualization
python3 demo_classifier.py
```

## 📈 Model Comparison

| Strategy | Algorithm | F1 Score | Rank |
|----------|-----------|----------|------|
| **BR** | **XGBoost** | **0.8933** | 🥇 **1st** |
| CC | XGBoost | 0.8916 | 🥈 2nd |
| BR | SVM | 0.8325 | 🥉 3rd |

## 🎨 Key Features

- ✅ **Multi-Label Classification**: Handle articles with multiple domains
- ✅ **High Performance**: 89.33% weighted F1 score
- ✅ **Fast Training**: 5.23 seconds training time
- ✅ **Easy Deployment**: Simple API for predictions
- ✅ **Comprehensive Testing**: Unit tests and validation
- ✅ **Detailed Reports**: Complete analysis and documentation
- ✅ **V0 Integration Ready**: Complete dashboard and demo package

## 📋 Requirements

- Python 3.13+
- pipenv
- 4GB RAM (recommended)
- 500MB disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🎨 V0 Visualization Package

The project includes a complete **V0 Visualization Package** for creating interactive dashboards and real-time classification demos:

### 📊 Dashboard Data
- **`dashboard_data.json`**: Complete project insights and metrics
- **`chart_data.json`**: Chart-ready data for V0 visualizations
- **Performance metrics**: F1 scores, accuracy, training time
- **Model comparison**: 9 different strategies tested
- **Dataset statistics**: 3,565 articles, label distribution

### 🤖 Real-time Classification
- **`real_time_classifier.py`**: Full-featured Python API
- **`demo_classifier.py`**: Standalone demo (no dependencies)
- **Single & batch classification** with confidence scores
- **Domain information** with icons and descriptions
- **Example articles** for testing
- **Mock predictions** when real model unavailable

### 🔌 API Integration
- **`api_endpoints.json`**: Complete API specification
- **Vercel deployment** ready
- **Error handling** and validation
- **Input/output schemas** for validation

### 🚀 Quick Start with V0
```bash
# Run the demo
cd v0_visualization
python3 demo_classifier.py

# View documentation
open v0_visualization/vo_documentation.md
```

For detailed V0 integration guide, see [`v0_visualization/README.md`](v0_visualization/README.md).

## 📄 License

This project is part of a machine learning challenge.

---

<div align="center">

**Developed by HealthCoders Lab Team 👩‍💻👨‍💻**

[![Report](https://img.shields.io/badge/📊-View%20Report-blue)](docs/model/model_documentation)
[![Results](https://img.shields.io/badge/📈-View%20Results-green)](output/)
[![V0 Demo](https://img.shields.io/badge/🎨-V0%20Demo-purple)](v0_visualization/)

</div>
