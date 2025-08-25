"""
Data processing module for medical article classification.
Handles data loading, cleaning, and preparation for multi-label classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for medical articles."""
    
    def __init__(self, file_path):
        """
        Initialize DataProcessor.
        
        Args:
            file_path (str): Path to the CSV file containing the dataset
        """
        self.file_path = file_path
        self.data = None
        self.mlb = MultiLabelBinarizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load the dataset from CSV file."""
        try:
            self.data = pd.read_csv(self.file_path, sep=';', encoding='utf-8')
            print(f"Dataset loaded successfully: {self.data.shape[0]} articles")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def process_labels(self, labels_str):
        """
        Process multi-label string into list of labels.
        
        Args:
            labels_str (str): String with labels separated by '|'
            
        Returns:
            list: List of individual labels
        """
        if pd.isna(labels_str):
            return []
        return labels_str.split('|')
    
    def prepare_data(self):
        """
        Prepare data for multi-label classification.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, label_names)
        """
        if self.data is None:
            self.load_data()
        
        # Clean text data
        print("Cleaning text data...")
        self.data['title_clean'] = self.data['title'].apply(self.clean_text)
        self.data['abstract_clean'] = self.data['abstract'].apply(self.clean_text)
        
        # Combine title and abstract
        self.data['text_combined'] = self.data['title_clean'] + ' ' + self.data['abstract_clean']
        
        # Process labels
        print("Processing labels...")
        labels_list = self.data['group'].apply(self.process_labels)
        
        # Transform labels to binary format
        y = self.mlb.fit_transform(labels_list)
        label_names = self.mlb.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['text_combined'], y, 
            test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        print(f"Number of labels: {len(label_names)}")
        print(f"Label names: {label_names}")
        
        return X_train, X_test, y_train, y_test, label_names
    
    def get_data_info(self):
        """Get information about the dataset."""
        if self.data is None:
            self.load_data()
        
        info = {
            'total_articles': len(self.data),
            'unique_titles': self.data['title'].nunique(),
            'unique_abstracts': self.data['abstract'].nunique(),
            'title_length_stats': {
                'mean': self.data['title'].str.len().mean(),
                'median': self.data['title'].str.len().median(),
                'min': self.data['title'].str.len().min(),
                'max': self.data['title'].str.len().max()
            },
            'abstract_length_stats': {
                'mean': self.data['abstract'].str.len().mean(),
                'median': self.data['abstract'].str.len().median(),
                'min': self.data['abstract'].str.len().min(),
                'max': self.data['abstract'].str.len().max()
            },
            'label_distribution': self.data['group'].value_counts()
        }
        
        return info 