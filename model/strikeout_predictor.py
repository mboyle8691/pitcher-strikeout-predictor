"""Main strikeout prediction model."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load
import os
from datetime import datetime
import sys
sys.path.append('..')
from config import MODEL_CHECKPOINTS, XGB_PARAMS, LGB_PARAMS


class StrikeoutPredictor:
    """Multi-model ensemble for strikeout predictions."""

    def __init__(self, model_type="ensemble", load_existing=True):
        """
        Initialize the strikeout predictor.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'ensemble'
            load_existing: Load existing model if available
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = None
        self.trained = False
        
        if load_existing:
            self.load_latest_model()

    def _create_models(self):
        """Create model instances."""
        if self.model_type in ["xgboost", "ensemble"]:
            self.models["xgb"] = xgb.XGBRegressor(**XGB_PARAMS)
        
        if self.model_type in ["lightgbm", "ensemble"]:
            self.models["lgb"] = lgb.LGBMRegressor(**LGB_PARAMS)
        
        if self.model_type == "ensemble":
            self.models["rf"] = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

    def prepare_features(self, X):
        """
        Prepare and scale features for modeling.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Processed feature array
        """
        X_processed = X.copy()
        
        # Encode categorical features
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col])
            else:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        # Scale features
        if "default" not in self.scalers:
            self.scalers["default"] = StandardScaler()
            X_processed = self.scalers["default"].fit_transform(X_processed)
        else:
            X_processed = self.scalers["default"].transform(X_processed)
        
        return X_processed

    def train(self, X, y, validation_split=0.1):
        """
        Train the model(s).
        
        Args:
            X: DataFrame with features
            y: Series with strikeout targets
            validation_split: Proportion for validation
        """
        self._create_models()
        self.feature_names = X.columns.tolist()
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train models
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            if model_name == "xgb":
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif model_name == "lgb":
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose_eval=False
                )
            else:
                model.fit(X_train, y_train)
        
        self.trained = True
        print("Model training complete.")
        self.save_model()

    def predict(self, X):
        """
        Make strikeout predictions.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Array of predicted strikeout counts or probabilities
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions.")
        
        X_processed = self.prepare_features(X)
        
        if self.model_type == "ensemble":
            # Average predictions from all models
            predictions = np.zeros(len(X))
            for model in self.models.values():
                predictions += model.predict(X_processed)
            return predictions / len(self.models)
        else:
            return list(self.models.values())[0].predict(X_processed)

    def predict_proba(self, X, threshold=None):
        """
        Get probability scores for strikeout predictions.
        
        Args:
            X: DataFrame with features
            threshold: Strikeout count threshold
            
        Returns:
            Probability scores
        """
        predictions = self.predict(X)
        if threshold:
            return (predictions >= threshold).astype(float)
        return predictions

    def save_model(self, name=None):
        """
        Save model to disk.
        
        Args:
            name: Custom model name (default: timestamp)
        """
        if not self.trained:
            return
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = MODEL_CHECKPOINTS / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            dump(model, save_path / f"{model_name}_model.joblib")
        
        # Save scalers and encoders
        dump(self.scalers, save_path / "scalers.joblib")
        dump(self.label_encoders, save_path / "label_encoders.joblib")
        dump(self.feature_names, save_path / "feature_names.joblib")
        
        print(f"Model saved to {save_path}")

    def load_latest_model(self):
        """
        Load the most recent saved model.
        """
        if not MODEL_CHECKPOINTS.exists():
            return
        
        checkpoints = sorted([d for d in MODEL_CHECKPOINTS.iterdir() if d.is_dir()], reverse=True)
        if not checkpoints:
            return
        
        latest = checkpoints[0]
        self._load_from_path(latest)

    def _load_from_path(self, path):
        """
        Load model from a specific path.
        
        Args:
            path: Path to model checkpoint
        """
        try:
            # Load models
            for model_file in path.glob("*_model.joblib"):
                model_name = model_file.stem.replace("_model", "")
                self.models[model_name] = load(model_file)
            
            # Load scalers and encoders
            self.scalers = load(path / "scalers.joblib")
            self.label_encoders = load(path / "label_encoders.joblib")
            self.feature_names = load(path / "feature_names.joblib")
            
            self.trained = True
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from trained models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.trained:
            return None
        
        importances = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
        
        # Average importance across models
        avg_importance = np.mean(list(importances.values()), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
