# Medical Article Classification Challenge

## Overview
This project implements a multi-label classification system for medical articles using traditional machine learning techniques. The system classifies articles into one or more domains: Cardiovascular, Neurological, Hepatorenal, and Oncological based on title and abstract content.

## Problem Statement
Develop an AI system capable of classifying medical articles into multiple domains using only title and abstract as inputs, applying various machine learning strategies and algorithms.

## Dataset
- **Size**: 3,565 articles
- **Format**: CSV with semicolon separator
- **Columns**: title, abstract, group (target)
- **Target**: Multi-label classification with 4 unique domains and 15 total categories (including combinations)

## Project Structure
```
model5/
├── input/
│   └── challenge_data.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── output/
│   ├── predictions.csv
│   └── results_comparison.csv
├── Pipfile
└── README.md
```

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pipenv install
   ```

2. **Activate virtual environment**:
   ```bash
   pipenv shell
   ```

3. **Run the main script**:
   ```bash
   pipenv run python src/main.py
   ```

## Usage

The project implements three multi-label classification strategies:
- Binary Relevance (BR)
- Classifier Chains (CC)
- Label Powerset (LP)

With three base algorithms:
- Logistic Regression
- XGBoost
- SVM

## Evaluation Metrics
- Hamming Loss
- Micro-F1 and Macro-F1
- Subset Accuracy

## Output
Results are saved in the `output/` directory:
- `predictions.csv`: Test dataset with predictions
- `results_comparison.csv`: Comparison table of all 9 model combinations

## Requirements
- Python 3.13
- See Pipfile for detailed dependencies

## Author
Data Scientist - Medical Article Classification Challenge 