# V0 Visualization & Real-time Classification Demo

This folder contains all the data and code needed to create interactive dashboards and real-time classification demos using V0 (Vercel).

## 📁 File Structure

```
v0_visualization/
├── 📊 dashboard_data.json          # Comprehensive project insights and metrics
├── 📈 chart_data.json              # Chart-ready data for V0 visualizations  
├── 🤖 real_time_classifier.py      # Real-time classification API
├── 🔌 api_endpoints.json           # API endpoint definitions
└── 📖 README.md                    # This file
```

## 🎯 Overview

This project implements a state-of-the-art **Binary Relevance + XGBoost** system for classifying medical articles into multiple domains:

- 🫀 **Cardiovascular** - Heart and blood vessel related research
- 🧠 **Neurological** - Brain and nervous system related research  
- 🫁 **Hepatorenal** - Liver and kidney related research
- 🦠 **Oncological** - Cancer and tumor related research

**Performance**: 89.33% Weighted F1 Score, 6.77% Hamming Loss

## 📊 Dashboard Data Files

### `dashboard_data.json`
Complete project insights including:
- **Performance metrics** (F1 scores, accuracy, training time)
- **Dataset statistics** (3,565 articles, label distribution)
- **Model comparison** (9 different strategies tested)
- **Feature engineering** details
- **Error analysis** and domain performance
- **Technical specifications**

### `chart_data.json` 
Ready-to-use chart data for V0 visualizations:
- **Performance charts** (radar, bar charts)
- **Label distribution** (pie charts, bar charts)
- **Text analysis** (length distributions)
- **Error analysis** (pie charts)
- **Domain performance** (radar charts)
- **Training progress** (comparison charts)
- **Feature importance** (bar charts)
- **Confusion matrix** data
- **Prediction examples**

## 🤖 Real-time Classification

### `real_time_classifier.py`
Complete Python API for real-time classification:

**Key Features:**
- ✅ **Single article classification**
- ✅ **Batch classification** 
- ✅ **Mock predictions** (fallback when model unavailable)
- ✅ **Domain information** with icons and descriptions
- ✅ **Example articles** for testing
- ✅ **Error handling** and validation

**Usage Example:**
```python
from real_time_classifier import RealTimeClassifier

# Initialize classifier
classifier = RealTimeClassifier()

# Classify single article
result = classifier.classify_single(
    title="Cardiac arrhythmia detection using machine learning",
    abstract="This study presents a novel approach for detecting cardiac arrhythmias..."
)

print(f"Predicted: {result['predicted_labels']}")
print(f"Confidence: {result['confidence']}")
```

**API Functions:**
- `api_classify_single(event)` - Single article classification
- `api_classify_batch(event)` - Batch classification
- `api_get_domains(event)` - Get domain information
- `api_get_examples(event)` - Get example articles
- `api_get_status(event)` - Get model status

### `api_endpoints.json`
Complete API specification including:
- **Endpoint definitions** with HTTP methods
- **Input/output schemas** for validation
- **Example requests/responses**
- **Integration guide** for Vercel deployment
- **Error handling** documentation

## 🚀 V0 Integration Guide

### 1. Dashboard Creation

Use the JSON data files to create interactive dashboards:

**Performance Overview:**
```javascript
// Load dashboard data
const dashboardData = await fetch('/api/dashboard-data').json();

// Create performance metrics cards
const f1Score = dashboardData.performance_metrics.primary_metrics.weighted_f1_score.value;
const hammingLoss = dashboardData.performance_metrics.primary_metrics.hamming_loss.value;
```

**Model Comparison Chart:**
```javascript
// Load chart data
const chartData = await fetch('/api/chart-data').json();

// Create radar chart
const radarData = chartData.performance_charts.model_comparison_radar;
// Use with Chart.js or similar library
```

### 2. Real-time Classification Demo

**Frontend Integration:**
```javascript
// Classify article
const response = await fetch('/api/classify-single', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "Cardiac arrhythmia detection using machine learning",
    abstract: "This study presents a novel approach..."
  })
});

const result = await response.json();
console.log(`Predicted: ${result.predicted_labels}`);
```

**Vercel API Routes:**
Create these files in your Vercel project:

```javascript
// api/classify-single.js
import { api_classify_single } from './real_time_classifier.py';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const result = api_classify_single(req.body);
  res.status(200).json(result);
}
```

### 3. Chart Integration

**Label Distribution Pie Chart:**
```javascript
const pieData = chartData.label_distribution_charts.single_labels_pie;
// Use with Chart.js, D3.js, or V0's chart components
```

**Performance Metrics Bar Chart:**
```javascript
const barData = chartData.performance_charts.metrics_bar_chart;
// Create responsive bar chart
```

## 📈 Key Insights for Visualization

### Performance Highlights
- **89.33% Weighted F1 Score** - Excellent performance
- **6.77% Hamming Loss** - Only 6.77% of label predictions are incorrect
- **5.23 seconds training time** - Fast model training
- **76.02% Subset Accuracy** - 76% of articles have all labels correctly predicted

### Dataset Characteristics
- **3,565 total articles** with title and abstract
- **4 medical domains** with multi-label classification
- **15 unique label combinations** (single and multiple labels)
- **Label imbalance** - neurological (950) vs oncological (215)

### Model Comparison Results
1. **BR + XGBoost** (89.33% F1) - Best performance
2. **CC + XGBoost** (89.16% F1) - Close second
3. **BR + SVM** (83.25% F1) - Good but slower

### Domain Performance
- **Neurological**: Excellent performance, clear domain separation
- **Cardiovascular**: High precision, some confusion with hepatorenal
- **Hepatorenal**: Good performance, some overlap with cardiovascular  
- **Oncological**: Strong performance, distinct medical terminology

## 🛠️ Technical Requirements

### Python Dependencies
```bash
pip install pandas numpy scikit-learn xgboost
```

### Vercel Deployment
1. Create API routes in `api/` directory
2. Upload model files to Vercel
3. Configure environment variables
4. Deploy with `vercel --prod`

### Frontend Requirements
- Modern browser with ES6+ support
- Chart.js, D3.js, or similar for visualizations
- Fetch API for HTTP requests

## 🎨 UI/UX Recommendations

### Color Scheme
- **Cardiovascular**: Red (#FF6B6B)
- **Neurological**: Blue (#4ECDC4) 
- **Hepatorenal**: Green (#45B7D1)
- **Oncological**: Purple (#96CEB4)

### Icons
- 🫀 Cardiovascular
- 🧠 Neurological
- 🫁 Hepatorenal  
- 🦠 Oncological

### Layout Suggestions
1. **Header**: Project title, performance summary
2. **Sidebar**: Navigation, model status
3. **Main Content**: Interactive charts and classification demo
4. **Footer**: Technical details, links

## 🔧 Customization

### Adding New Charts
1. Add data to `chart_data.json`
2. Create corresponding API endpoint
3. Implement frontend visualization

### Extending Classification
1. Modify `real_time_classifier.py`
2. Add new domain descriptions
3. Update API schemas

### Performance Monitoring
1. Add logging to classification functions
2. Track prediction accuracy
3. Monitor API response times

## 📞 Support

For questions or issues:
1. Check the comprehensive report in `../output/COMPREHENSIVE_REPORT.md`
2. Review the main project README
3. Test with example articles provided

---

**Ready to build amazing interactive dashboards with V0! 🚀**
