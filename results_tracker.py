"""Track actual results and compare with predictions."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys

sys.path.append('utils')

from config import RESULTS_DIR, MODEL_HISTORY_DIR, PREDICTION_OUTPUT


class ResultsTracker:
    """Track results and compare with predictions."""

    def __init__(self):
        """Initialize results tracker."""
        self.results_df = None
        self.predictions_df = None
        self.performance_log = MODEL_HISTORY_DIR / "daily_performance.csv"

    def load_results(self, results_path: str) -> pd.DataFrame:
        """
        Load daily results from file.
        
        Args:
            results_path: Path to results CSV
            
        Returns:
            DataFrame with results
        """
        self.results_df = pd.read_csv(results_path)
        print(f"Loaded {len(self.results_df)} results from {results_path}")
        return self.results_df

    def load_predictions(self, predictions_path: str = None) -> pd.DataFrame:
        """
        Load predictions from file.
        
        Args:
            predictions_path: Path to predictions CSV
            
        Returns:
            DataFrame with predictions
        """
        if predictions_path:
            self.predictions_df = pd.read_csv(predictions_path)
        else:
            # Load most recent predictions
            prediction_files = sorted(PREDICTION_OUTPUT.glob('*.csv'), reverse=True)
            if prediction_files:
                self.predictions_df = pd.read_csv(prediction_files[0])
        
        print(f"Loaded {len(self.predictions_df)} predictions")
        return self.predictions_df

    def compare_predictions_to_results(self) -> pd.DataFrame:
        """
        Compare predictions with actual results.
        
        Returns:
            DataFrame with comparison metrics
        """
        if self.predictions_df is None or self.results_df is None:
            raise ValueError("Must load both predictions and results first.")
        
        # Merge on pitcher identifier
        merged = pd.merge(
            self.predictions_df,
            self.results_df,
            left_on='pitcher_id',
            right_on='pitcher_id',
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        # Calculate metrics
        merged['strikeout_error'] = merged['actual_strikeouts'] - merged['predicted_strikeouts']
        merged['absolute_error'] = np.abs(merged['strikeout_error'])
        merged['percent_error'] = (merged['strikeout_error'] / (merged['actual_strikeouts'] + 1)) * 100
        
        # Hit/Miss for over/under
        merged['over_under_pred'] = (merged['predicted_strikeouts'] >= 6.5).astype(int)  # Over/Under threshold
        merged['over_under_actual'] = (merged['actual_strikeouts'] >= 6.5).astype(int)
        merged['over_under_hit'] = (merged['over_under_pred'] == merged['over_under_actual']).astype(int)
        
        return merged

    def calculate_metrics(self, comparison_df: pd.DataFrame) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            comparison_df: DataFrame from compare_predictions_to_results
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'num_predictions': len(comparison_df),
            'mean_absolute_error': comparison_df['absolute_error'].mean(),
            'rmse': np.sqrt((comparison_df['strikeout_error'] ** 2).mean()),
            'mean_prediction': comparison_df['predicted_strikeouts'].mean(),
            'mean_actual': comparison_df['actual_strikeouts'].mean(),
            'prediction_bias': comparison_df['strikeout_error'].mean(),
            'over_under_accuracy': comparison_df['over_under_hit'].mean(),
            'predictions_high': (comparison_df['predicted_strikeouts'] > comparison_df['actual_strikeouts']).sum(),
            'predictions_low': (comparison_df['predicted_strikeouts'] < comparison_df['actual_strikeouts']).sum(),
            'perfect_predictions': (comparison_df['strikeout_error'] == 0).sum(),
        }
        
        return metrics

    def log_daily_performance(self, metrics: dict):
        """
        Log daily performance metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        log_df = pd.DataFrame([metrics])
        
        if self.performance_log.exists():
            existing = pd.read_csv(self.performance_log)
            log_df = pd.concat([existing, log_df], ignore_index=True)
        
        log_df.to_csv(self.performance_log, index=False)
        print(f"Performance logged to {self.performance_log}")

    def display_metrics(self, metrics: dict):
        """
        Display performance metrics in formatted output.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("DAILY PERFORMANCE METRICS")
        print("="*60)
        print(f"Timestamp:           {metrics['timestamp']}")
        print(f"Predictions Made:    {metrics['num_predictions']}")
        print(f"\nAccuracy Metrics:")
        print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.2f} strikeouts")
        print(f"  RMSE:               {metrics['rmse']:.2f}")
        print(f"  Over/Under Hit:     {metrics['over_under_accuracy']:.1%}")
        print(f"\nBias Analysis:")
        print(f"  Mean Predicted:     {metrics['mean_prediction']:.2f}")
        print(f"  Mean Actual:        {metrics['mean_actual']:.2f}")
        print(f"  Prediction Bias:    {metrics['prediction_bias']:+.2f}")
        print(f"  Predicted High:     {metrics['predictions_high']} times")
        print(f"  Predicted Low:      {metrics['predictions_low']} times")
        print(f"\nPerfect Predictions: {metrics['perfect_predictions']}")
        print("="*60 + "\n")

    def save_detailed_comparison(self, comparison_df: pd.DataFrame) -> str:
        """
        Save detailed comparison to file.
        
        Args:
            comparison_df: DataFrame from compare_predictions_to_results
            
        Returns:
            Path to saved file
        """
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = RESULTS_DIR / filename
        
        comparison_df.to_csv(filepath, index=False)
        print(f"Detailed comparison saved to {filepath}")
        
        return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Track and analyze strikeout prediction results")
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--predictions', type=str, help='Path to predictions CSV file')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ResultsTracker()
    
    # Load data
    tracker.load_results(args.results)
    tracker.load_predictions(args.predictions)
    
    # Compare
    print("\nComparing predictions to results...")
    comparison = tracker.compare_predictions_to_results()
    
    # Calculate metrics
    metrics = tracker.calculate_metrics(comparison)
    
    # Display metrics
    tracker.display_metrics(metrics)
    
    # Log performance
    tracker.log_daily_performance(metrics)
    
    # Save detailed comparison
    tracker.save_detailed_comparison(comparison)
    
    print("Results tracking complete.")


if __name__ == "__main__":
    main()
