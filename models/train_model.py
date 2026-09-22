"""
Script to train, evaluate, and serialize the Fake News Detection Machine Learning Model.
"""

import os
import sys

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.preprocessing import train_fake_news_classifier


def main():
    data_path = os.path.join(BASE_DIR, "data", "fake_news.csv")
    model_save_path = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
    
    print("=" * 60)
    print("Training Fake News Detection ML Classifier (TF-IDF + Logistic Regression)")
    print("=" * 60)
    print(f"Loading training data from: {data_path}")
    
    pipeline, metrics = train_fake_news_classifier(
        data_path=data_path,
        model_save_path=model_save_path
    )
    
    print(f"\n[+] Model Training Complete!")
    print(f"[+] Model saved to: {model_save_path}")
    print(f"[+] Total Training Samples: {metrics['train_size']}")
    print(f"[+] Total Test Samples: {metrics['test_size']}")
    print(f"[+] Model Test Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
    
    print("-" * 40)
    print("Confusion Matrix [REAL, FAKE]:")
    print(metrics['confusion_matrix'])
    print("-" * 40)
    print("\nClassification Report:")
    for cls, scores in metrics['report'].items():
        if isinstance(scores, dict):
            print(f"  Class '{cls}': Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1-score']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
