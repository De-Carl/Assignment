import pandas as pd
import numpy as np
import joblib
import os
import sys
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import top_k_accuracy_score

from data_preprocessor import DataPreprocessor


def load_artifacts(artifacts_dir):
    """Load model and all required encoders"""
    artifacts = {}
    required_files = ['model', 'mlb_encoder', 'ohe_encoder', 'scaler', 'target_encoder']
    
    print(f"Loading model components from {artifacts_dir}...")
    try:
        for name in required_files:
            path = os.path.join(artifacts_dir, f'{name}.joblib')
            artifacts[name] = joblib.load(path)
        return artifacts
    except FileNotFoundError as e:
        print(f"Error: File {e.filename} not found. Please run model_trainer.py first.")
        return None


def engineer_features(df, artifacts):
    """
    Reproduce the feature engineering logic from model_trainer.py.
    Must be consistent with training process.
    """
    print("Building feature matrix...")
    
    # Time features
    df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    df = df.dropna(subset=['posted_date']).copy()

    df['post_year'] = df['posted_date'].dt.year
    df['post_month'] = df['posted_date'].dt.month
    df['day_of_week'] = df['posted_date'].dt.dayofweek
    
    # Time index calculation
    min_date = df['posted_date'].min()
    df['time_index'] = (df['posted_date'] - min_date).dt.days

    # Define column names (consistent with training code)
    numerical_cols = ['salary_avg_usd', 'post_year', 'post_month', 'day_of_week', 'time_index']
    categorical_cols = ['employment_type_standardized', 'experience_level_standardized', 'country']
    skills_col = 'skills_list_processed'
    target_col = 'job_title_standardized'

    # Filter missing values
    required_cols = numerical_cols + categorical_cols + [skills_col, target_col]
    df_clean = df.dropna(subset=required_cols).copy()
    
    # Filter unknown target labels
    known_labels = set(artifacts['target_encoder'].classes_)
    initial_len = len(df_clean)
    df_clean = df_clean[df_clean[target_col].isin(known_labels)]
    if len(df_clean) < initial_len:
        print(f"Warning: Filtered {initial_len - len(df_clean)} records with unknown job titles.")

    print(f"Valid test samples: {len(df_clean)}")

    # Feature encoding (transform only, using trained mappings)
    X_skills = artifacts['mlb_encoder'].transform(df_clean[skills_col])
    X_cat = artifacts['ohe_encoder'].transform(df_clean[categorical_cols])
    X_num = artifacts['scaler'].transform(df_clean[numerical_cols])
    
    # Combine feature matrix
    X = hstack([X_skills, X_cat, csr_matrix(X_num)])
    
    # Encode target variable
    y = artifacts['target_encoder'].transform(df_clean[target_col])
    
    return X, y, df_clean

def evaluate_accuracy(X, y, artifacts):
    """Perform predictions and calculate Top-1 and Top-5 accuracy"""
    model = artifacts['model']
    target_encoder = artifacts['target_encoder']
    
    print("Running batch predictions...")
    y_prob = model.predict_proba(X)
    
    # Calculate Top-K accuracy
    acc_top1 = top_k_accuracy_score(y, y_prob, k=1)
    acc_top5 = top_k_accuracy_score(y, y_prob, k=5)
    
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"Total test samples: {X.shape[0]}")
    print("-" * 50)
    print(f"Top-1 Accuracy (exact match):      {acc_top1:.2%}")
    print(f"Top-5 Accuracy (in top 5 preds):   {acc_top5:.2%}")
    print("=" * 50)

    # Show prediction examples
    print("\nPrediction Examples (5 random samples):")
    sample_indices = np.random.choice(X.shape[0], 5, replace=False)
    top5_indices_all = np.argsort(y_prob, axis=1)[:, -5:][:, ::-1]
    
    for i, idx in enumerate(sample_indices, 1):
        true_label_idx = y[idx]
        true_label_name = target_encoder.inverse_transform([true_label_idx])[0]
        
        pred_label_indices = top5_indices_all[idx]
        pred_label_names = list(target_encoder.inverse_transform(pred_label_indices))
        
        # Find the rank position (1-based) if hit
        if true_label_name in pred_label_names:
            rank = pred_label_names.index(true_label_name) + 1
            result = f"Correct! (Rank {rank})"
        else:
            result = "Incorrect (Not in Top-5)"
        
        print(f"\nSample {i}:")
        print(f"  Actual: {true_label_name}")
        print(f"  Top-5 Predictions: {', '.join(pred_label_names)}")
        print(f"  Result: {result}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, '..', 'dataset', 'ai_job_market_unified.csv')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    
    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    preprocessor = DataPreprocessor(file_path=dataset_path)
    df = preprocessor.run_preprocessing()
    
    if df is None:
        return

    # Load model artifacts
    artifacts = load_artifacts(artifacts_dir)
    if artifacts is None:
        return

    # Feature engineering
    X, y, df_clean = engineer_features(df, artifacts)

    # Evaluate model
    evaluate_accuracy(X, y, artifacts)


if __name__ == "__main__":
    main()