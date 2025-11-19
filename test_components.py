import pandas as pd
from data_manager import load_data, preprocess_data
from feature_selector import select_features
from model_trainer import train_evaluate

def test_backend():
    print("Testing Data Manager...")
    df, error = load_data()
    if error:
        print(f"Error loading data: {error}")
        return
    print(f"Data loaded. Shape: {df.shape}")
    
    X, y = preprocess_data(df)
    print(f"Data preprocessed. X shape: {X.shape}, y shape: {y.shape}")

    print("\nTesting Feature Selector...")
    methods = ['RFE (Recursive Feature Elimination)', 'Mutual Information', 'LASSO (L1 Regularization)']
    for method in methods:
        print(f"Testing {method}...")
        selected, scores = select_features(method, X, y, k=3)
        print(f"Selected features: {selected}")
        print(f"Scores: {scores}")

    print("\nTesting Model Trainer...")
    models = ['Logistic Regression', 'Random Forest']
    for model_name in models:
        print(f"Testing {model_name}...")
        results = train_evaluate(model_name, X, y)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    test_backend()
