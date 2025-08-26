# Medical Article Classification - BR + XGBoost Model

## Overview
This project implements a Binary Relevance + XGBoost multi-label classification system for medical articles. The system classifies articles into one or more domains: Cardiovascular, Neurological, Hepatorenal, or Oncological based on title and abstract content.

## Project Structure
```
model12/
├── input/                 # Input data
model12/
├── input/                 # Input data
│   └── challenge_data.csv
├── models/               # Trained models
│   ├── BR_xgboost_model.pkl
│   └── feature_pipeline.pkl
├── output/               # Results and visualizations
├── src_clean/           # Simplified source code
│   ├── data/            # Data processing
│   ├── features/        # Feature engineering
│   ├── models/          # BR + XGBoost model
│   ├── train_model.py   # Training pipeline
│   └── test_model.py    # Testing script
├── tests_clean/         # Unit tests
├── Pipfile              # Dependencies
└── README_CLEAN.md     # This file
```

## Model Performance
- **Strategy**: Binary Relevance (BR)
- **Algorithm**: XGBoost
- **Weighted F1 Score**: 0.8933
- **Hamming Loss**: 0.0677
- **Subset Accuracy**: 0.7602

## Setup Instructions

### Prerequisites
- Python 3.13
- pipenv

### Installation
1. Install dependencies:
   ```bash
   pipenv install
   ```

### Usage

#### Training the Model
```bash
pipenv run python src_clean/train_model.py
```

#### Testing with New Data
```bash
pipenv run python src_clean/test_model.py
```

#### Running Tests
```bash
pipenv run pytest tests_clean/ -v
```

## Model Testing Examples

### Single Article
```python
from src_clean.test_model import ModelTester

tester = ModelTester()
result = tester.predict_single(
    title="Cardiac arrhythmia detection using machine learning",
    abstract="This study presents a novel approach for detecting cardiac arrhythmias..."
)
print(f"Predicted labels: {result['predicted_labels']}")
```

### Multiple Articles
```python
titles = ["Title 1", "Title 2", "Title 3"]
abstracts = ["Abstract 1", "Abstract 2", "Abstract 3"]

results = tester.predict_multiple(titles, abstracts)
for result in results:
    print(f"Title: {result['title']}")
    print(f"Labels: {result['predicted_labels']}")
```

## Output Labels
The model predicts one or more of these labels:
- `cardiovascular`
- `neurological` 
- `hepatorenal`
- `oncological`

**Output format**:
- Single label: `"cardiovascular"`
- Multiple labels: `"cardiovascular|neurological"`
- No labels: `"none"`

## Dependencies
- **Data Processing**: pandas, numpy, scikit-learn
- **Machine Learning**: xgboost
- **Text Processing**: nltk
- **Testing**: pytest

## License
This project is part of a machine learning challenge.
