"""
Feature engineering module for medical article classification.
Handles text vectorization and feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Handles feature engineering for text data."""
    
    def __init__(self, max_features=5000, n_components=1000):
        """
        Initialize FeatureEngineer.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            n_components (int): Number of components for dimensionality reduction
        """
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf_vectorizer = None
        self.feature_pipeline = None
        
    def create_tfidf_features(self, X_train, X_test=None, fit=True):
        """
        Create TF-IDF features from text data.
        
        Args:
            X_train (pd.Series): Training text data
            X_test (pd.Series): Test text data (optional)
            fit (bool): Whether to fit the vectorizer on training data
            
        Returns:
            tuple: (X_train_tfidf, X_test_tfidf) or (X_train_tfidf, None)
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
            print(f"TF-IDF features created: {X_train_tfidf.shape[1]} features")
            
            if X_test is not None:
                X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
                return X_train_tfidf, X_test_tfidf
            else:
                return X_train_tfidf, None
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Set fit=True first.")
            X_train_tfidf = self.tfidf_vectorizer.transform(X_train)
            if X_test is not None:
                X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
                return X_train_tfidf, X_test_tfidf
            else:
                return X_train_tfidf, None
    
    def create_dimensionality_reduction_pipeline(self, X_train, X_test=None):
        """
        Create a pipeline with TF-IDF and dimensionality reduction.
        
        Args:
            X_train (pd.Series): Training text data
            X_test (pd.Series): Test text data (optional)
            
        Returns:
            tuple: (X_train_reduced, X_test_reduced) or (X_train_reduced, None)
        """
        # Create pipeline with TF-IDF and SVD
        self.feature_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('svd', TruncatedSVD(n_components=self.n_components, random_state=42))
        ])
        
        # Fit and transform training data
        X_train_reduced = self.feature_pipeline.fit_transform(X_train)
        print(f"Reduced features created: {X_train_reduced.shape[1]} features")
        
        if X_test is not None:
            X_test_reduced = self.feature_pipeline.transform(X_test)
            return X_train_reduced, X_test_reduced
        else:
            return X_train_reduced, None
    
    def get_feature_names(self):
        """Get feature names from the TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            return None
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """
        Get top features by TF-IDF score.
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names
        """
        if self.tfidf_vectorizer is None:
            return None
        
        feature_names = self.get_feature_names()
        if feature_names is None:
            return None
        
        # Get feature scores (mean TF-IDF values)
        feature_scores = np.mean(self.tfidf_vectorizer.idf_)
        top_indices = np.argsort(feature_scores)[-n:]
        
        return [feature_names[i] for i in top_indices]
    
    def create_custom_features(self, X_train, X_test=None):
        """
        Create custom features for medical text.
        
        Args:
            X_train (pd.Series): Training text data
            X_test (pd.Series): Test text data (optional)
            
        Returns:
            tuple: (X_train_custom, X_test_custom) or (X_train_custom, None)
        """
        def extract_medical_features(text):
            """Extract medical-specific features from text."""
            features = {}
            
            # Medical terminology indicators
            medical_terms = [
                'cardiac', 'heart', 'cardiovascular', 'brain', 'neurological',
                'liver', 'kidney', 'hepatorenal', 'cancer', 'oncology',
                'tumor', 'metastasis', 'therapy', 'treatment', 'diagnosis'
            ]
            
            text_lower = text.lower()
            for term in medical_terms:
                features[f'has_{term}'] = 1 if term in text_lower else 0
            
            # Text length features
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # Medical abbreviations
            medical_abbrevs = ['ct', 'mri', 'ecg', 'eeg', 'cbc', 'bun', 'creatinine']
            for abbrev in medical_abbrevs:
                features[f'has_{abbrev}'] = 1 if abbrev in text_lower else 0
            
            return features
        
        # Extract features for training data
        train_features = [extract_medical_features(text) for text in X_train]
        X_train_custom = pd.DataFrame(train_features)
        
        if X_test is not None:
            test_features = [extract_medical_features(text) for text in X_test]
            X_test_custom = pd.DataFrame(test_features)
            return X_train_custom, X_test_custom
        else:
            return X_train_custom, None 