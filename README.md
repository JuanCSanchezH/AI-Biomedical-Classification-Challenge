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
cd model12

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
model12/
├── 🗂️ input/                    # Dataset
│   └── challenge_data.csv
├── 🤖 models/                   # Trained models
│   ├── BR_xgboost_model.pkl
│   └── feature_pipeline.pkl
├── 📊 output/                   # Results & reports
│   ├── comprehensive_report.png
│   └── COMPREHENSIVE_REPORT.md
├── 💻 src/                      # Source code
│   ├── data/loader.py
│   ├── features/vectorizer.py
│   ├── models/br_xgboost_model.py
│   ├── train_model.py
│   └── test_model.py
└── 🧪 tests/                    # Unit tests
    └── test_br_xgboost_model.py
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

## 📄 License

This project is part of a machine learning challenge.

---

<div align="center">

**Developed by HealthCoders Lab Team 👩‍💻👨‍💻**

[![Report](https://img.shields.io/badge/📊-View%20Report-blue)](output/COMPREHENSIVE_REPORT.md)
[![Results](https://img.shields.io/badge/📈-View%20Results-green)](output/)

</div>
