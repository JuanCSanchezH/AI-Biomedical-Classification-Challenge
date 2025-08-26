# 📊 Enhanced CSV Testing Guide

## 🎯 Overview

The enhanced `test_model.py` now supports CSV file upload with comprehensive metrics and confusion matrix generation. This guide explains how to use the new functionality.

## 📋 CSV File Requirements

### Required Columns
Your CSV file must contain these columns:
- `title`: Article titles
- `abstract`: Article abstracts  
- `group`: True labels (for evaluation)

### CSV Format Examples

#### Semicolon Separated (Recommended)
```csv
title;abstract;group
Cardiac arrhythmia detection;This study presents...;cardiovascular
Brain imaging analysis;This research investigates...;neurological
Liver function study;We examined liver function...;cardiovascular|hepatorenal
```

#### Comma Separated
```csv
title,abstract,group
Cardiac arrhythmia detection,This study presents...,cardiovascular
Brain imaging analysis,This research investigates...,neurological
Liver function study,We examined liver function...,cardiovascular|hepatorenal
```

## 🚀 Usage

### 1. Basic CSV Testing
```python
from src.test_model import EnhancedModelTester

# Initialize tester
tester = EnhancedModelTester()

# Test with CSV file
results = tester.predict_csv(
    csv_path="your_data.csv",
    output_path="output/results.csv"
)
```

### 2. Command Line Usage
```bash
pipenv run python src/test_model.py
```

## 📊 Output Files

### 1. Results CSV
- **File**: `output/csv_test_results.csv`
- **Columns**: Original data + `group_predicted`
- **Format**: Semicolon separated

### 2. Confusion Matrix
- **File**: `output/csv_test_results_confusion_matrix.png`
- **Content**: Per-label confusion matrices
- **Format**: High-resolution PNG

## 📈 Metrics Provided

### Main Metric: Weighted F1 Score
- **Formula**: (Micro F1 + Macro F1) / 2
- **Range**: 0-1 (higher is better)
- **Example**: 0.9028

### Supporting Metrics
- **Micro F1 Score**: Overall precision and recall
- **Macro F1 Score**: Average F1 across labels
- **Subset Accuracy**: Exact match percentage
- **Hamming Loss**: Label-wise error rate
- **Per-Label F1**: Individual label performance

## 🔍 Example Results

### Sample Output
```
==================================================
EVALUATION METRICS
==================================================
Weighted F1 Score (Main Metric): 0.9028
Micro F1 Score: 0.8889
Macro F1 Score: 0.9167
Subset Accuracy: 0.7500
Hamming Loss: 0.0625

Per-Label F1 Scores:
  cardiovascular: 0.6667
  hepatorenal: 1.0000
  neurological: 1.0000
  oncological: 1.0000
==================================================
```

### Results CSV Structure
```csv
title;abstract;group;group_predicted
Cardiac arrhythmia detection;This study presents...;cardiovascular;cardiovascular
Brain imaging analysis;This research investigates...;neurological;neurological
Liver function study;We examined liver function...;cardiovascular|hepatorenal;hepatorenal
```

## 🎨 Confusion Matrix Features

### Visualization
- **4 Subplots**: One for each label (cardiovascular, hepatorenal, neurological, oncological)
- **Color Coding**: Blue heatmap with darker = higher values
- **Annotations**: Actual numbers in each cell
- **Accuracy**: Per-label accuracy displayed

### Interpretation
- **True Positives**: Diagonal elements (top-left to bottom-right)
- **False Positives**: Off-diagonal elements in columns
- **False Negatives**: Off-diagonal elements in rows
- **Accuracy**: (TP + TN) / Total for each label

## 🛠️ Advanced Usage

### Custom Model Paths
```python
tester = EnhancedModelTester(
    model_path="custom/path/model.pkl",
    feature_pipeline_path="custom/path/pipeline.pkl"
)
```

### Single Article Testing
```python
result = tester.predict_single(
    title="Your article title",
    abstract="Your article abstract"
)
print(f"Predicted: {result['predicted_labels']}")
```

### Multiple Articles Testing
```python
titles = ["Title 1", "Title 2", "Title 3"]
abstracts = ["Abstract 1", "Abstract 2", "Abstract 3"]

results = tester.predict_multiple(titles, abstracts)
for result in results:
    print(f"Labels: {result['predicted_labels']}")
```

## ⚠️ Important Notes

### Data Quality
- Ensure CSV encoding is UTF-8
- Handle missing values appropriately
- Validate label format (use '|' for multiple labels)

### Performance
- Large CSV files may take longer to process
- Memory usage scales with file size
- Consider batch processing for very large datasets

### Error Handling
- Missing columns will raise ValueError
- Invalid label formats may affect metrics
- File not found errors are handled gracefully

## 📝 Troubleshooting

### Common Issues
1. **Missing Columns**: Ensure title, abstract, group columns exist
2. **Encoding Issues**: Use UTF-8 encoding for CSV files
3. **Separator Problems**: Try both semicolon and comma separators
4. **Memory Issues**: Process large files in batches

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

tester = EnhancedModelTester()
results = tester.predict_csv("your_file.csv")
```

## 🎯 Best Practices

1. **Data Preparation**: Clean and validate your CSV data
2. **Label Consistency**: Use consistent label naming
3. **File Organization**: Keep input and output files organized
4. **Backup**: Always backup original data before processing
5. **Validation**: Cross-check results with domain experts

---

**Happy Testing! 🚀**
