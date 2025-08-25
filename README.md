# Medical Article Classification Challenge

## Overview
This project implements a multi-label classification system for medical articles using machine learning techniques. The system classifies articles into one or more domains: Cardiovascular, Neurological, Hepatorenal, or Oncological based on title and abstract content.

## Problem Statement
Develop an AI system capable of classifying medical articles into multiple domains using only title and abstract as inputs, applying traditional machine learning techniques, natural language models, or hybrid approaches.

## Dataset
- **Size**: 3,565 articles
- **Format**: CSV with columns: title, abstract, group (target)
- **Target**: Multi-label classification with 4 unique domains
- **Features**: Title and abstract text

## Project Structure
```
model12/
├── input/                 # Input data
│   └── challenge_data.csv
├── models/               # Trained models and model code
├── output/               # Results, charts, and predictions
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   ├── features/        # Feature engineering
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit tests
├── Pipfile              # Dependencies
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Setup Instructions

### Prerequisites
- Python 3.13
- pipenv

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pipenv install
   ```

### Usage
1. Activate the environment:
   ```bash
   pipenv shell
   ```

2. Run the main pipeline:
   ```bash
   pipenv run python src/main.py
   ```

3. Run tests:
   ```bash
   pipenv run pytest
   ```

## Model Strategies
The project implements three multi-label classification strategies:
- **Binary Relevance (BR)**: Treats each label independently
- **Classifier Chains (CC)**: Uses label dependencies
- **Label Powerset (LP)**: Treats each label combination as a class

## Base Algorithms
- Logistic Regression
- XGBoost
- Support Vector Machine (SVM)

## Evaluation Metrics
- Hamming Loss
- Micro-F1 and Macro-F1
- Subset Accuracy
- Weighted F1 Score

## Output
Results are saved in the `output/` folder including:
- Model performance charts
- Comparison tables
- Confusion matrices
- Predictions on test data

## Code Quality
- Follows PEP8 standards
- Uses ruff for linting and formatting
- Pre-commit hooks for code quality
- Comprehensive test coverage

## License
This project is part of a machine learning challenge. 