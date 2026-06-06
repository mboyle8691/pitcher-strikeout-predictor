"""Model training pipeline."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
sys.path.append('..')
from strikeout_predictor import StrikeoutPredictor
from config import MODEL_HISTORY_DIR
from datetime import datetime


class TrainingPipeline:
    """Pipeline for training and evaluating strikeout models."""

    def __init__(self, model_type="ensemble"):
        """
        Initialize training pipeline.
        
        Args:
            model_type: Type of model to train
        """
        self.model = StrikeoutPredictor(model_type=model_type, load_existing=False)
        self.training_history = []
        self.model_type = model_type

    def train_model(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.1):
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (strikeout counts)
            validation_split: Proportion of data to use for validation
        """
        print(f"Training {self.model_type} model...")
        self.model.train(X, y, validation_split=validation_split)
        print("Training complete.")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
        """
        Perform cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV metrics
        """
        print(f"Running {cv_folds}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.model.models.get('xgb', self.model.models.get('rf')),
            self.model.prepare_features(X),
            y,
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )
        
        metrics = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"CV R² Score: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")
        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        }
        
        print("\n=== Model Performance ===")
        print(f"R² Score:  {metrics['r2_score']:.4f}")
        print(f"RMSE:      {metrics['rmse']:.4f}")
        print(f"MAE:       {metrics['mae']:.4f}")
        print(f"MAPE:      {metrics['mape']:.2f}%")
        print("======================\n")
        
        return metrics

    def log_performance(self, metrics: dict, model_name: str = None):
        """
        Log model performance to file.
        
        Args:
            metrics: Dictionary of performance metrics
            model_name: Optional name for model checkpoint
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'model_type': self.model_type,
            **metrics
        }
        
        log_df = pd.DataFrame([log_entry])
        
        log_file = MODEL_HISTORY_DIR / "performance_log.csv"
        if log_file.exists():
            existing = pd.read_csv(log_file)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        
        log_df.to_csv(log_file, index=False)
        print(f"Performance logged to {log_file}")

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        return self.model.get_feature_importance(top_n=top_n)
