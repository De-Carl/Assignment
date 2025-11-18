import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import joblib
import os
import warnings

# Import SMOTE (Synthetic Minority Over-sampling TEchnique)
# This library is required to handle class imbalance.
# You may need to install it: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Warning: 'imbalanced-learn' library not found. SMOTE functionality will be unavailable.")
    print("Please install it via: pip install imbalanced-learn")
    SMOTE = None

# Ensure data_preprocessor.py is in the same directory or Python path
try:
    from data_preprocessor import DataPreprocessor
except ImportError:
    print("Warning: DataPreprocessor could not be imported. This script assumes it's run")
    print("as __main__ where DataPreprocessor is defined or available.")


class AdvancedModelTrainer:
    """
    AdvancedModelTrainer (V2)
    
    This class handles the end-to-end training pipeline, incorporating:
    1.  Addition of 'location' as a key predictive feature.
    2.  Use of SMOTE to address severe class imbalance.
    3.  Generation of a 'salary_map' artifact for (job, location) -> median_salary lookups.
    4.  Evaluation using Top-K accuracy metrics.
    """
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.artifacts = {}
        self.output_dir = 'artifacts'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _create_time_features(self):
        """
        Engineers time-based features from the 'posted_date' column.
        """
        print("Creating time-series features...")
        self.df['posted_date'] = pd.to_datetime(self.df['posted_date'], errors='coerce')
        self.df.dropna(subset=['posted_date'], inplace=True)

        self.df['post_year'] = self.df['posted_date'].dt.year
        self.df['post_month'] = self.df['posted_date'].dt.month
        self.df['day_of_week'] = self.df['posted_date'].dt.dayofweek
        
        # Create a numerical time index (days since the first post)
        min_date = self.df['posted_date'].min()
        self.df['time_index'] = (self.df['posted_date'] - min_date).dt.days
        print("Time-series features created: year, month, day_of_week, time_index.")

    def _prepare_data(self):
        """
        Prepares the feature matrix (X) and target vector (y).
        
        This V2 implementation explicitly adds 'location' to the categorical features,
        as analysis showed it is a highly predictive feature.
        """
        self._create_time_features()

        self.numerical_features_cols = [
            'salary_avg_usd',  
            'post_year', 
            'post_month', 
            'day_of_week', 
            'time_index'
        ]
        
        # --- V2 CORE MODIFICATION: Add 'location' ---
        # Based on data analysis, 'location' is a high-importance feature.
        # data_preprocessor.py ensures this column is loaded.
        self.categorical_features_cols = [
            'employment_type_standardized', 
            'experience_level_standardized',
            'country'  # Added feature
        ]
        
        self.skills_col = 'skills_list_processed'
        self.target = 'job_title_standardized'
         
        # Check for the existence of all required columns
        all_feature_cols = self.numerical_features_cols + self.categorical_features_cols + [self.skills_col, self.target]
        
        existing_cols = [col for col in all_feature_cols if col in self.df.columns]
        missing_cols = [col for col in all_feature_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: The following columns are missing and will be skipped: {missing_cols}")
            
        # Update feature lists to only use columns that actually exist
        self.numerical_features_cols = [col for col in self.numerical_features_cols if col in existing_cols]
        self.categorical_features_cols = [col for col in self.categorical_features_cols if col in existing_cols]
        
        if self.skills_col not in existing_cols:
            print(f"Error: Critical 'skills' column ({self.skills_col}) is missing! Aborting.")
            return False
        
        # Drop rows where any of the critical features or target are missing
        self.df.dropna(subset=existing_cols, inplace=True)
        
        self.X_df = self.df[self.numerical_features_cols + self.categorical_features_cols + [self.skills_col]]
        self.y = self.df[self.target]
        
        print(f"Data preparation complete. Training with {self.df.shape[0]} samples.")
        return True

    def _encode_features(self):
        """
        Encodes all features using the appropriate transformers.
        - Skills: MultiLabelBinarizer
        - Categorical: OneHotEncoder (now includes 'location')
        - Numerical: StandardScaler
        """
        print("Encoding features...")
        
        # 1. Encode skills (Multi-label)
        self.artifacts['mlb_encoder'] = MultiLabelBinarizer(sparse_output=True)
        skills_encoded = self.artifacts['mlb_encoder'].fit_transform(self.X_df[self.skills_col])

        # 2. Encode categorical features (One-hot)
        # This will now automatically handle 'location' as it's in the list
        self.artifacts['ohe_encoder'] = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        categoricals_encoded = self.artifacts['ohe_encoder'].fit_transform(self.X_df[self.categorical_features_cols])

        # 3. Scale numerical features
        self.artifacts['scaler'] = StandardScaler()
        numerical_features_scaled = self.artifacts['scaler'].fit_transform(self.X_df[self.numerical_features_cols])
        
        # 4. Combine all encoded features into a single sparse matrix
        self.X_encoded = hstack([
            skills_encoded, 
            categoricals_encoded, 
            csr_matrix(numerical_features_scaled)
        ])
        print(f"Final feature matrix shape: {self.X_encoded.shape}")
        
        # 5. Encode target variable
        self.artifacts['target_encoder'] = LabelEncoder()
        self.y_encoded = self.artifacts['target_encoder'].fit_transform(self.y)
        print("Feature and target encoding complete.")
        
    def _create_salary_map(self):
        """
        (New in V2)
        Creates a lookup dictionary (map) for (job_title, location) -> median_salary.
        This artifact is saved for use in the prediction phase.
        """
        if 'country' not in self.df.columns or 'salary_avg_usd' not in self.df.columns:
            print("Error: Cannot create salary map. 'country' column is missing.")
            return

        # We use median() to get a robust measure of central tendency,
        # which is less sensitive to salary outliers than mean().
        salary_map = self.df.groupby(
            ['job_title_standardized', 'country']
        )['salary_avg_usd'].median().to_dict()
        
        self.artifacts['salary_map'] = salary_map
        print(f"Salary map for (job, country) -> median_salary created. (Total entries: {len(salary_map)})")
    def _analyze_feature_importance(self, model):
        """
        (New in V2)
        Analyzes and prints the top feature importances from the trained model.
        This helps verify that 'skills' and 'location' are key drivers.
        """
        print("\n--- Feature Importance Analysis ---")
        try:
            # Retrieve all feature names from the encoders
            mlb_features = self.artifacts['mlb_encoder'].classes_
            ohe_features = self.artifacts['ohe_encoder'].get_feature_names_out(self.categorical_features_cols)
            num_features = np.array(self.numerical_features_cols)
            
            # Ensure all names are strings before concatenation
            feature_names = np.concatenate([
                mlb_features.astype(str), 
                ohe_features.astype(str), 
                num_features.astype(str)
            ])
            
            importances = model.feature_importances_
            
            # Create a DataFrame for easy sorting
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            
            print("Top 25 most important features:")
            print(importance_df.head(25).to_string())

            # Check total importance of 'location' features
            location_features = [f for f in ohe_features if 'location_' in f]
            location_importance = importance_df[importance_df['feature'].isin(location_features)]['importance'].sum()
            print(f"\nTotal importance of all 'location' features: {location_importance:.2f}")

        except Exception as e:
            print(f"Could not compute feature importance: {e}")

    def train(self):
        """
        Executes the full training pipeline:
        1. Prepares data (calling _prepare_data)
        2. Encodes features (calling _encode_features)
        3. Splits data into train/test sets
        4. Applies SMOTE to the training set to fix imbalance
        5. Runs RandomizedSearchCV for hyperparameter tuning
        6. Evaluates the best model using Top-K accuracy
        7. Creates the salary map (calling _create_salary_map)
        """
        if not self._prepare_data():
            print("Data preparation failed. Terminating training.")
            return
            
        self._encode_features()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_encoded, self.y_encoded,
            test_size=0.2, random_state=42, stratify=self.y_encoded
        )
        
        print("\n--- Starting Job Title Classifier Training (V2: SMOTE + Location) ---")
        
        # --- V2 CORE MODIFICATION: Apply SMOTE ---
        # Given the severe class imbalance (e.g., 19% vs 0.7%),
        # SMOTE will create synthetic samples for the minority classes.
        if SMOTE is None:
            print("SMOTE is not available. Training without resampling.")
            X_train_resampled, y_train_resampled = X_train, y_train
        else:
            print("Applying SMOTE to handle class imbalance...")
            print(f"Original training set size: {X_train.shape[0]}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"Resampled training set size: {X_train_resampled.shape[0]}")
        
        
        print("Starting hyperparameter tuning with RandomizedSearchCV...")
        param_dist = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 50, 70],
            'max_depth': [10, 15, 20],
        }

        # We do not use class_weight='balanced' because SMOTE has already
        # rebalanced the training data. Using both can over-correct.
        lgbm = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        
        random_search = RandomizedSearchCV(
            lgbm, param_distributions=param_dist, n_iter=10, # 10 iterations for speed
            cv=3, n_jobs=-1, verbose=2, random_state=42,
            scoring='f1_weighted'
        )
        
        # Fit on the *resampled* data
        random_search.fit(X_train_resampled, y_train_resampled)
        
        print(f"\nBest parameters found: {random_search.best_params_}")
        best_model = random_search.best_estimator_
        
        self.artifacts['model'] = best_model
        print("--- Model Training & Tuning Complete ---")

        # --- V2 EVALUATION: Report Top-K Accuracy ---
        # Evaluate on the original, non-resampled test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) # Get probabilities for Top-K

        # Calculate Top-K metrics
        accuracy_top1 = accuracy_score(y_test, y_pred)
        accuracy_top3 = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        accuracy_top5 = top_k_accuracy_score(y_test, y_pred_proba, k=5)

        print(f"\n=== V2 Model Performance (on unbalanced test set) ===")
        print(f"Top-1 Accuracy: {accuracy_top1:.4f}")
        print(f"Top-3 Accuracy: {accuracy_top3:.4f}")
        print(f"Top-5 Accuracy: {accuracy_top5:.4f}")

        # Full classification report
        target_names = self.artifacts['target_encoder'].classes_
        print("\nFull Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # --- Run new V2 methods ---
        self._analyze_feature_importance(best_model)
        self._create_salary_map()
        
    def save_artifacts(self):
        """
        Saves all generated artifacts (model, encoders, scaler, and salary_map)
        to the 'artifacts' directory using joblib.
        """
        if not self.artifacts:
            print("No artifacts to save. Please run training first.")
            return

        print("\n--- Saving Artifacts ---")
        for name, artifact in self.artifacts.items():
            file_path = os.path.join(self.output_dir, f'{name}.joblib')
            joblib.dump(artifact, file_path)
            print(f"Saved {name} to {file_path}")
        print("--- Artifacts Saved Successfully ---")

if __name__ == '__main__':
    # 1. Load and preprocess data
    # We explicitly pass the file path to ensure it finds the correct CSV
    preprocessor = DataPreprocessor(file_path='..\\dataset\\ai_job_market_unified.csv')
    processed_df = preprocessor.run_preprocessing()

    if processed_df is not None:
        # 2. Initialize and run the V2 trainer
        trainer_v2 = AdvancedModelTrainer(processed_df)
        trainer_v2.train()
        
        # 3. Save all artifacts (model, encoders, salary_map)
        trainer_v2.save_artifacts()