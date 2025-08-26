#!/usr/bin/env python3
"""
Simplified Medical Article Classification Demo for V0
This script provides a standalone demo without external dependencies.
"""

import json
from typing import Dict, List, Any


class DemoClassifier:
    """
    Simplified classifier for medical articles that can be used with V0.
    Provides mock predictions for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the demo classifier."""
        
        # Medical domain descriptions for better UX
        self.domain_descriptions = {
            "cardiovascular": {
                "name": "Cardiovascular",
                "description": "Heart and blood vessel related research",
                "icon": "🫀",
                "keywords": ["cardiac", "heart", "cardiovascular", "artery", "vein", "blood pressure", "arrhythmia", "myocardial"]
            },
            "neurological": {
                "name": "Neurological", 
                "description": "Brain and nervous system related research",
                "icon": "🧠",
                "keywords": ["brain", "neural", "neurological", "stroke", "alzheimer", "parkinson", "cerebral", "cognitive"]
            },
            "hepatorenal": {
                "name": "Hepatorenal",
                "description": "Liver and kidney related research", 
                "icon": "🫁",
                "keywords": ["liver", "kidney", "hepatorenal", "renal", "hepatitis", "dialysis", "nephrology", "hepatic"]
            },
            "oncological": {
                "name": "Oncological",
                "description": "Cancer and tumor related research",
                "icon": "🦠", 
                "keywords": ["cancer", "tumor", "oncology", "chemotherapy", "malignant", "carcinoma", "metastasis", "oncology"]
            }
        }
    
    def _mock_prediction(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        Generate mock predictions based on keyword matching.
        
        Args:
            title: Article title
            abstract: Article abstract
            
        Returns:
            Mock prediction results
        """
        # Simple keyword-based mock prediction
        text = (title + " " + abstract).lower()
        
        predictions = {
            "cardiovascular": 0,
            "neurological": 0, 
            "hepatorenal": 0,
            "oncological": 0
        }
        
        # Mock prediction logic based on keywords
        if any(word in text for word in ["cardiac", "heart", "cardiovascular", "artery", "blood", "arrhythmia", "myocardial"]):
            predictions["cardiovascular"] = 1
            
        if any(word in text for word in ["brain", "neural", "stroke", "alzheimer", "parkinson", "cerebral", "cognitive"]):
            predictions["neurological"] = 1
            
        if any(word in text for word in ["liver", "kidney", "renal", "hepatitis", "dialysis", "nephrology", "hepatic"]):
            predictions["hepatorenal"] = 1
            
        if any(word in text for word in ["cancer", "tumor", "oncology", "chemotherapy", "malignant", "carcinoma", "metastasis"]):
            predictions["oncological"] = 1
        
        # If no predictions, default to neurological
        if sum(predictions.values()) == 0:
            predictions["neurological"] = 1
        
        # Convert to label format
        predicted_labels = [k for k, v in predictions.items() if v == 1]
        label_string = "|".join(predicted_labels) if predicted_labels else "none"
        
        # Calculate mock confidence based on keyword matches
        total_keywords = 0
        matched_keywords = 0
        
        for domain, keywords in self.domain_descriptions.items():
            total_keywords += len(keywords["keywords"])
            for keyword in keywords["keywords"]:
                if keyword in text:
                    matched_keywords += 1
        
        confidence = min(0.95, max(0.70, matched_keywords / max(1, total_keywords) * 2))
        
        return {
            "title": title,
            "abstract": abstract,
            "predicted_labels": label_string,
            "prediction_matrix": list(predictions.values()),
            "label_names": list(predictions.keys()),
            "confidence": round(confidence, 3),
            "is_mock": True,
            "matched_keywords": matched_keywords,
            "total_keywords_checked": total_keywords
        }
    
    def classify_single(self, title: str, abstract: str) -> Dict[str, Any]:
        """
        Classify a single medical article.
        
        Args:
            title: Article title
            abstract: Article abstract
            
        Returns:
            Classification results with predictions and metadata
        """
        if not title or not abstract:
            return {
                "error": "Title and abstract are required",
                "status": "error"
            }
        
        try:
            result = self._mock_prediction(title, abstract)
            
            # Add domain information
            result["domains"] = {}
            for label in result["label_names"]:
                if label in self.domain_descriptions:
                    result["domains"][label] = self.domain_descriptions[label]
            
            result["status"] = "success"
            return result
            
        except Exception as e:
            return {
                "error": f"Classification failed: {str(e)}",
                "status": "error"
            }
    
    def classify_batch(self, articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Classify multiple medical articles.
        
        Args:
            articles: List of dictionaries with 'title' and 'abstract' keys
            
        Returns:
            Batch classification results
        """
        if not articles:
            return {
                "error": "Articles list is required",
                "status": "error"
            }
        
        try:
            results = []
            for article in articles:
                title = article.get("title", "")
                abstract = article.get("abstract", "")
                result = self._mock_prediction(title, abstract)
                
                # Add domain information
                result["domains"] = {}
                for label in result["label_names"]:
                    if label in self.domain_descriptions:
                        result["domains"][label] = self.domain_descriptions[label]
                
                results.append(result)
            
            return {
                "results": results,
                "total_articles": len(results),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": f"Batch classification failed: {str(e)}",
                "status": "error"
            }
    
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about the medical domains.
        
        Returns:
            Domain descriptions and metadata
        """
        return {
            "domains": self.domain_descriptions,
            "total_domains": len(self.domain_descriptions),
            "status": "success"
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get the current status of the model.
        
        Returns:
            Model status information
        """
        return {
            "is_loaded": True,
            "model_type": "Demo Classifier (Mock Predictions)",
            "description": "Keyword-based classification for demonstration purposes",
            "status": "success"
        }
    
    def get_example_articles(self) -> List[Dict[str, str]]:
        """
        Get example articles for testing.
        
        Returns:
            List of example articles
        """
        return [
            {
                "title": "Cardiac arrhythmia detection using machine learning algorithms",
                "abstract": "This study presents a novel approach for detecting cardiac arrhythmias using machine learning techniques. We analyzed electrocardiogram data from 500 patients and developed a classification system that can identify various types of arrhythmias with high accuracy."
            },
            {
                "title": "Neural network analysis of brain imaging data for Alzheimer's detection",
                "abstract": "This research investigates the use of neural networks for analyzing brain imaging data to detect early signs of Alzheimer's disease. We used MRI scans from 300 patients and achieved 88% accuracy in early detection."
            },
            {
                "title": "Liver function tests in patients with cardiovascular disease",
                "abstract": "We examined liver function in 150 patients with cardiovascular disease. Results showed significant correlations between cardiac markers and liver enzyme levels, suggesting potential organ interactions."
            },
            {
                "title": "Novel chemotherapy agents for lung cancer treatment",
                "abstract": "A new chemotherapy protocol was tested on 200 lung cancer patients. The treatment showed 40% improvement in survival rates compared to standard chemotherapy regimens."
            },
            {
                "title": "Stroke rehabilitation using virtual reality technology",
                "abstract": "Virtual reality technology was implemented for stroke rehabilitation in 80 patients. The system showed 60% improvement in motor function recovery compared to traditional rehabilitation methods."
            },
            {
                "title": "Renal function assessment in diabetic patients",
                "abstract": "This study evaluated kidney function in 250 diabetic patients using advanced imaging techniques and blood markers. Results indicate early detection of renal complications."
            },
            {
                "title": "Brain-computer interface for motor rehabilitation",
                "abstract": "A novel brain-computer interface system was developed for motor rehabilitation in stroke patients. The system achieved 75% accuracy in decoding motor intentions."
            },
            {
                "title": "Cardiovascular risk assessment using artificial intelligence",
                "abstract": "AI-based risk assessment model for cardiovascular diseases was developed using patient data from 1000 individuals. The model achieved 90% accuracy in predicting cardiac events."
            }
        ]


# API endpoints for V0 integration
def api_classify_single(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint for single article classification.
    
    Args:
        event: Dictionary containing 'title' and 'abstract'
        
    Returns:
        Classification results
    """
    classifier = DemoClassifier()
    
    title = event.get("title", "")
    abstract = event.get("abstract", "")
    
    return classifier.classify_single(title, abstract)


def api_classify_batch(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint for batch article classification.
    
    Args:
        event: Dictionary containing 'articles' list
        
    Returns:
        Batch classification results
    """
    classifier = DemoClassifier()
    
    articles = event.get("articles", [])
    
    return classifier.classify_batch(articles)


def api_get_domains(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint to get domain information.
    
    Args:
        event: Empty dictionary (not used)
        
    Returns:
        Domain information
    """
    classifier = DemoClassifier()
    return classifier.get_domain_info()


def api_get_examples(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint to get example articles.
    
    Args:
        event: Empty dictionary (not used)
        
    Returns:
        Example articles
    """
    classifier = DemoClassifier()
    examples = classifier.get_example_articles()
    
    return {
        "examples": examples,
        "total_examples": len(examples),
        "status": "success"
    }


def api_get_status(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint to get model status.
    
    Args:
        event: Empty dictionary (not used)
        
    Returns:
        Model status information
    """
    classifier = DemoClassifier()
    return classifier.get_model_status()


# Demo function for testing
def demo():
    """Run a demonstration of the classifier."""
    print("🏥 Medical Article Classification Demo")
    print("=" * 50)
    
    classifier = DemoClassifier()
    
    # Get example articles
    examples = classifier.get_example_articles()
    
    print(f"\n📚 Testing with {len(examples)} example articles:")
    print("-" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Abstract: {example['abstract'][:100]}...")
        
        # Classify
        result = classifier.classify_single(example['title'], example['abstract'])
        
        if result['status'] == 'success':
            print(f"   🎯 Predicted: {result['predicted_labels']}")
            print(f"   📊 Confidence: {result.get('confidence', 'N/A')}")
            print(f"   🔍 Keywords matched: {result.get('matched_keywords', 'N/A')}")
            print("   ⚠️  (Demo prediction)")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    
    # Show domain information
    print("\n🏷️  Medical Domains:")
    print("-" * 30)
    domain_info = classifier.get_domain_info()
    for domain_key, domain_data in domain_info['domains'].items():
        print(f"{domain_data['icon']} {domain_data['name']}: {domain_data['description']}")


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo()
