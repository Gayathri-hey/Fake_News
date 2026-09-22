"""
Text preprocessing and Machine Learning classification utilities for Fake News Detection.
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text: str) -> str:
    """
    Preprocesses raw text:
    1. Converts text to lowercase
    2. Removes URLs and HTML tags
    3. Removes punctuation and special characters
    4. Removes extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_fake_news_classifier(data_path: str, model_save_path: str = None):
    """
    Trains a Logistic Regression model with TF-IDF vectorization on the provided dataset.
    
    Returns:
        pipeline: Trained Scikit-Learn Pipeline
        metrics: Dictionary containing accuracy, report, and confusion matrix
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validation of columns
    req_cols = {'title', 'text', 'label'}
    if not req_cols.issubset(df.columns):
        # If columns slightly differ (e.g., lowercase / uppercase)
        df.columns = [c.strip().lower() for c in df.columns]
        if not {'text', 'label'}.issubset(df.columns):
            raise ValueError(f"Dataset must contain 'text' (or 'title') and 'label' columns.")
    
    # Combine title and text if title exists
    if 'title' in df.columns:
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    else:
        df['combined_text'] = df['text'].fillna('')
        
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Filter empty rows
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Standardize labels to FAKE and REAL
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[df['label'].isin(['FAKE', 'REAL'])]
    
    if len(df) < 4:
        raise ValueError("Dataset has too few samples to train a model.")
    
    X = df['cleaned_text']
    y = df['label']
    
    # Train-test split (stratified if possible)
    stratify_target = y if len(df['label'].value_counts()) > 1 and df['label'].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_target
    )
    
    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
    
    metrics = {
        'accuracy': float(acc),
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'test_size': len(y_test),
        'train_size': len(X_train),
        'classes': list(pipeline.classes_)
    }
    
    # Save model if path specified
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(pipeline, model_save_path)
        
    return pipeline, metrics


def load_model(model_path: str):
    """Loads a pre-trained model pipeline from disk."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_news(pipeline, text: str):
    """
    Predicts whether a given news text is FAKE or REAL.
    
    Returns:
        prediction (str): 'FAKE' or 'REAL'
        confidence (float): Probability of the predicted class (0.0 to 1.0)
        probabilities (dict): Map of class name to probability
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 'UNKNOWN', 0.0, {'FAKE': 0.5, 'REAL': 0.5}
    
    pred = pipeline.predict([cleaned])[0]
    probs = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
    confidence = float(prob_dict.get(pred, 0.5))
    
    return pred, confidence, prob_dict
