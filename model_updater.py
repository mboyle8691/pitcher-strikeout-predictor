"""Retrain model with new historical data."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append('model')
sys.path.append('utils')

from training import TrainingPipeline
from features import FeatureEngineer
from data_processor import DataProcessor
from config import DATA_DIR, MODEL_HISTORY_DIR


class ModelUpdater:
    """Update and retrain model with historical data."""

    def __init__(self):
        """Initialize model updater."""
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.training_pipeline = None

    def collect_training_data(self, data_dir: Path = None) -> pd.DataFrame:
        """
        Collect and aggregate training data from historical results.
        
        Args:
            data_dir: Directory containing result files
            
        Returns:
            Aggregated DataFrame with all historical data
        """
        if data_dir is None:
            data_dir = DATA_DIR / "results"
        
        all_data = []
        
        for csv_file in data_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if not all_data:
            raise ValueError(f"No data found in {data_dir}")
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records collected: {len(combined)}")
        
        return combined

    def prepare_training_data(self, raw_data: pd.DataFrame) -> tuple:
        """
        Prepare data for training.
        
        Args:
            raw_data: Raw historical data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Remove duplicates
        data = raw_data.drop_duplicates()
        
        # Filter out incomplete records
        required_cols = ['actual_strikeouts', 'pitcher_id']
        data = data.dropna(subset=required_cols)
        
        print(f"Records after cleaning: {len(data)}")
        
        # Extract target
        y = data['actual_strikeouts'].astype(float)
        
        # Build features
        X_list = []
        
        for idx, row in data.iterrows():
            try:
                # Pitcher stats
                pitcher_features = self.feature_engineer.create_pitcher_stats_features(row)
                
                # Handedness
                pitcher_hand = row.get('pitcher_handedness', 'R')
                batter_hand = row.get('batter_handedness', 'R')
                handedness_features = self.feature_engineer.create_handedness_features(
                    pitcher_hand, batter_hand, row.to_dict()
                )
                
                # Situational
                situational_features = self.feature_engineer.create_situational_features(
                    inning=row.get('inning', 1),
                    outs=row.get('outs', 0),
                    runners_on=[
                        row.get('runner_first', False),
                        row.get('runner_second', False),
                        row.get('runner_third', False)
                    ],
                    score_diff=row.get('score_diff', 0),
                    home_away='H' if row.get('home_game', True) else 'A'
                )
                
                # Team
                team_features = self.feature_engineer.create_team_features(row.to_dict())
                
                # Combine
                combined = self.feature_engineer.combine_features(
                    pitcher_features, handedness_features, situational_features, team_features
                )
                X_list.append(combined)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        X = pd.concat(X_list, ignore_index=True)
        
        # Handle missing values
        X = self.feature_engineer.handle_missing_values(X, strategy='median')
        
        # Create derived features
        X = self.feature_engineer.create_derived_features(X)
        
        print(f"Features prepared: {X.shape}")
        print(f"Target prepared: {y.shape}")
        
        return X, y

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, model_type: str = "ensemble"):
        """
        Train model and evaluate performance.
        
        Args:
            X: Features
            y: Target
            model_type: Type of model to train
        """
        # Initialize training pipeline
        self.training_pipeline = TrainingPipeline(model_type=model_type)
        
        # Train model
        self.training_pipeline.train_model(X, y, validation_split=0.15)
        
        # Cross-validate
        cv_metrics = self.training_pipeline.cross_validate(X, y, cv_folds=5)
        
        # Split data for final evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Evaluate
        eval_metrics = self.training_pipeline.evaluate(X_test, y_test)
        
        # Log performance
        all_metrics = {**cv_metrics, **eval_metrics}
        self.training_pipeline.log_performance(all_metrics, model_name=model_type)
        
        # Feature importance
        importance_df = self.training_pipeline.get_feature_importance(top_n=20)
        print("\n=== Top 20 Feature Importance ===")
        print(importance_df.to_string())
        print("\n")
        
        return eval_metrics

    def generate_model_report(self):
        """Generate comprehensive model report."""
        if not MODEL_HISTORY_DIR.exists():
            print("No model history found.")
            return
        
        # Load performance log
        log_file = MODEL_HISTORY_DIR / "performance_log.csv"
        if log_file.exists():
            performance_log = pd.read_csv(log_file)
            print("\n" + "="*60)
            print("MODEL PERFORMANCE HISTORY")
            print("="*60)
            print(performance_log.tail(10).to_string())
            print("="*60 + "\n")
        
        # Load daily performance
        daily_log = MODEL_HISTORY_DIR / "daily_performance.csv"
        if daily_log.exists():
            daily_perf = pd.read_csv(daily_log)
            print("\n" + "="*60)
            print("DAILY PREDICTION PERFORMANCE")
            print("="*60)
            print(daily_perf.tail(10).to_string())
            print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update and retrain strikeout model")
    parser.add_argument('--data-dir', type=str, help='Directory with result files')
    parser.add_argument('--model-type', type=str, default='ensemble', 
                       choices=['xgboost', 'lightgbm', 'ensemble'],
                       help='Type of model to train')
    parser.add_argument('--report', action='store_true', help='Generate model report')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = ModelUpdater()
    
    if args.report:
        updater.generate_model_report()
        return
    
    # Collect training data
    print("Collecting historical training data...")
    raw_data = updater.collect_training_data(
        Path(args.data_dir) if args.data_dir else None
    )
    
    # Prepare data
    print("\nPreparing training data...")
    X, y = updater.prepare_training_data(raw_data)
    
    # Train and evaluate
    print(f"\nTraining {args.model_type} model...")
    metrics = updater.train_and_evaluate(X, y, model_type=args.model_type)
    
    print("Model update complete.")


if __name__ == "__main__":
    main()
