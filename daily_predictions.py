"""Generate daily strikeout predictions from lineup."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import sys

sys.path.append('model')
sys.path.append('utils')

from strikeout_predictor import StrikeoutPredictor
from features import FeatureEngineer
from data_processor import DataProcessor
from config import PREDICTION_OUTPUT, LINEUPS_DIR


class DailyPredictionGenerator:
    """Generate daily strikeout predictions."""

    def __init__(self):
        """Initialize prediction generator."""
        self.predictor = StrikeoutPredictor(load_existing=True)
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()

    def load_lineup(self, lineup_path: str) -> pd.DataFrame:
        """
        Load daily lineup from file.
        
        Args:
            lineup_path: Path to lineup CSV
            
        Returns:
            DataFrame with lineup data
        """
        lineup = pd.read_csv(lineup_path)
        print(f"Loaded {len(lineup)} pitchers from {lineup_path}")
        return lineup

    def generate_predictions(self, lineup: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for daily lineup.
        
        Args:
            lineup: DataFrame with pitcher/game information
            
        Returns:
            DataFrame with predictions
        """
        predictions_list = []
        
        for idx, row in lineup.iterrows():
            try:
                # Extract pitcher information
                pitcher_id = row.get('pitcher_id')
                pitcher_name = row.get('pitcher_name')
                team = row.get('team')
                opponent = row.get('opponent')
                
                # Build feature vector
                features = self._build_feature_vector(row)
                
                if features is not None:
                    # Generate prediction
                    pred = self.predictor.predict(features)[0]
                    
                    predictions_list.append({
                        'pitcher_id': pitcher_id,
                        'pitcher_name': pitcher_name,
                        'team': team,
                        'opponent': opponent,
                        'predicted_strikeouts': pred,
                        'confidence': self._get_confidence(pred),
                        'prediction_category': self._categorize_prediction(pred),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                print(f"Error processing pitcher {row.get('pitcher_name')}: {e}")
        
        predictions_df = pd.DataFrame(predictions_list)
        return predictions_df.sort_values('predicted_strikeouts', ascending=False)

    def _build_feature_vector(self, row: pd.Series) -> pd.DataFrame:
        """
        Build feature vector from lineup row.
        
        Args:
            row: Single lineup row
            
        Returns:
            DataFrame with features or None if data incomplete
        """
        try:
            # Pitcher stats features
            pitcher_features = self.feature_engineer.create_pitcher_stats_features(row)
            
            # Handedness features (assuming avg batter handedness for team)
            pitcher_hand = row.get('pitcher_handedness', 'R')
            batter_hand = row.get('avg_batter_handedness', 'R')
            handedness_features = self.feature_engineer.create_handedness_features(
                pitcher_hand, batter_hand, row.to_dict()
            )
            
            # Situational features
            situational_features = self.feature_engineer.create_situational_features(
                inning=1,  # Starting prediction
                outs=0,
                runners_on=[False, False, False],
                score_diff=0,
                home_away='H' if row.get('home_game', True) else 'A'
            )
            
            # Team features
            team_features = self.feature_engineer.create_team_features(
                row.get('team_data', {})
            )
            
            # Combine all features
            all_features = self.feature_engineer.combine_features(
                pitcher_features, handedness_features, situational_features, team_features
            )
            
            # Handle missing values
            all_features = self.feature_engineer.handle_missing_values(all_features)
            
            # Create derived features
            all_features = self.feature_engineer.create_derived_features(all_features)
            
            return all_features
        
        except Exception as e:
            print(f"Error building features: {e}")
            return None

    def _get_confidence(self, prediction: float) -> str:
        """
        Determine confidence level of prediction.
        
        Args:
            prediction: Predicted strikeout count
            
        Returns:
            Confidence category
        """
        if prediction >= 8:
            return "High"
        elif prediction >= 5:
            return "Medium"
        else:
            return "Low"

    def _categorize_prediction(self, prediction: float) -> str:
        """
        Categorize prediction into buckets.
        
        Args:
            prediction: Predicted strikeout count
            
        Returns:
            Category string
        """
        if prediction >= 10:
            return "Elite"
        elif prediction >= 8:
            return "Very Good"
        elif prediction >= 6:
            return "Good"
        elif prediction >= 4:
            return "Average"
        else:
            return "Below Average"

    def save_predictions(self, predictions: pd.DataFrame, output_dir: Path = None) -> str:
        """
        Save predictions to file.
        
        Args:
            predictions: DataFrame with predictions
            output_dir: Directory to save to
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = PREDICTION_OUTPUT
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = output_dir / filename
        
        predictions.to_csv(filepath, index=False)
        print(f"\nPredictions saved to {filepath}")
        
        return str(filepath)

    def display_predictions(self, predictions: pd.DataFrame, top_n: int = 15):
        """
        Display top predictions in formatted table.
        
        Args:
            predictions: DataFrame with predictions
            top_n: Number of predictions to display
        """
        print("\n" + "="*100)
        print(f"TOP {min(top_n, len(predictions))} STRIKEOUT PREDICTIONS")
        print("="*100)
        
        display_df = predictions.head(top_n)[[
            'pitcher_name', 'team', 'opponent', 'predicted_strikeouts', 
            'confidence', 'prediction_category'
        ]].copy()
        
        display_df['predicted_strikeouts'] = display_df['predicted_strikeouts'].round(2)
        
        print(display_df.to_string(index=False))
        print("="*100 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate daily strikeout predictions")
    parser.add_argument('--lineup', type=str, help='Path to lineup CSV file')
    parser.add_argument('--display', type=int, default=15, help='Number of predictions to display')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DailyPredictionGenerator()
    
    # Load lineup
    if args.lineup:
        lineup_path = args.lineup
    else:
        # Look for most recent lineup file
        lineup_files = sorted(LINEUPS_DIR.glob('*.csv'), reverse=True)
        if not lineup_files:
            print("No lineup files found. Please provide a lineup CSV.")
            return
        lineup_path = lineup_files[0]
    
    lineup = generator.load_lineup(lineup_path)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generator.generate_predictions(lineup)
    
    # Save predictions
    saved_path = generator.save_predictions(predictions)
    
    # Display top predictions
    generator.display_predictions(predictions, top_n=args.display)
    
    print(f"Total predictions generated: {len(predictions)}")


if __name__ == "__main__":
    main()
