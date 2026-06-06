"""Feature engineering for strikeout predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FeatureEngineer:
    """Generate and engineer features for strikeout prediction."""

    @staticmethod
    def create_pitcher_stats_features(pitcher_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from pitcher statistics.
        
        Args:
            pitcher_data: DataFrame with pitcher stats
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Basic strikeout metrics
        features['k_per_9'] = pitcher_data.get('k_per_9', 0)
        features['k_per_bb'] = pitcher_data.get('strikeouts', 0) / (pitcher_data.get('walks', 1))
        
        # Performance metrics
        features['era'] = pitcher_data.get('era', 0)
        features['whip'] = pitcher_data.get('whip', 0)
        features['fip'] = pitcher_data.get('fip', 0)
        
        # Velocity and movement
        features['avg_velocity'] = pitcher_data.get('avg_velocity', 0)
        features['spin_rate'] = pitcher_data.get('spin_rate', 0)
        features['induced_vb'] = pitcher_data.get('induced_vb', 0)  # induced vertical break
        
        # Recent performance
        features['last_10_k9'] = pitcher_data.get('last_10_k9', features['k_per_9'])
        features['last_10_era'] = pitcher_data.get('last_10_era', features['era'])
        
        # Workload
        features['innings_pitched'] = pitcher_data.get('innings_pitched', 0)
        features['games_started'] = pitcher_data.get('games_started', 0)
        features['games_appeared'] = pitcher_data.get('games_appeared', 0)
        
        return features

    @staticmethod
    def create_handedness_features(
        pitcher_handedness: str,
        batter_handedness: str,
        pitcher_stats: Dict
    ) -> pd.DataFrame:
        """
        Create features based on pitcher/batter handedness matchups.
        
        Args:
            pitcher_handedness: 'L' or 'R'
            batter_handedness: 'L' or 'R'
            pitcher_stats: Dictionary of pitcher statistics
            
        Returns:
            DataFrame with handedness features
        """
        features = pd.DataFrame(index=[0])
        
        # Handedness encoding
        features['pitcher_right_handed'] = 1 if pitcher_handedness == 'R' else 0
        features['batter_right_handed'] = 1 if batter_handedness == 'R' else 0
        
        # Matchup type
        matchup = f"{pitcher_handedness}/{batter_handedness}"
        matchup_type_map = {'L/L': 0, 'L/R': 1, 'R/L': 2, 'R/R': 3}
        features['matchup_type'] = matchup_type_map.get(matchup, -1)
        
        # Handedness advantage
        same_handed = pitcher_handedness == batter_handedness
        features['same_handed_matchup'] = 1 if same_handed else 0
        
        # Handedness-specific K/9
        handedness_key = f"k9_vs_{batter_handedness}"
        features['k9_vs_batter_hand'] = pitcher_stats.get(handedness_key, pitcher_stats.get('k_per_9', 0))
        
        return features

    @staticmethod
    def create_matchup_features(
        pitcher_id: str,
        batter_id: str,
        historical_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create features from historical pitcher vs batter matchups.
        
        Args:
            pitcher_id: Pitcher identifier
            batter_id: Batter identifier
            historical_data: Historical matchup data
            
        Returns:
            DataFrame with matchup features
        """
        features = pd.DataFrame(index=[0])
        
        if historical_data is not None and len(historical_data) > 0:
            matchup = historical_data[
                (historical_data['pitcher_id'] == pitcher_id) & 
                (historical_data['batter_id'] == batter_id)
            ]
            
            if len(matchup) > 0:
                features['pitcher_vs_batter_strikeout_pct'] = matchup['strikeout_pct'].mean()
                features['pitcher_vs_batter_at_bats'] = len(matchup)
            else:
                features['pitcher_vs_batter_strikeout_pct'] = 0
                features['pitcher_vs_batter_at_bats'] = 0
        else:
            features['pitcher_vs_batter_strikeout_pct'] = 0
            features['pitcher_vs_batter_at_bats'] = 0
        
        return features

    @staticmethod
    def create_situational_features(
        inning: int,
        outs: int,
        runners_on: List[bool],
        score_diff: int,
        home_away: str
    ) -> pd.DataFrame:
        """
        Create game situation features.
        
        Args:
            inning: Current inning (1-9)
            outs: Number of outs (0-2)
            runners_on: [first, second, third]
            score_diff: Runs difference (negative=down, positive=up)
            home_away: 'H' for home, 'A' for away
            
        Returns:
            DataFrame with situational features
        """
        features = pd.DataFrame(index=[0])
        
        features['inning'] = inning
        features['outs'] = outs
        features['runners_on_base'] = sum(runners_on)
        features['bases_loaded'] = 1 if all(runners_on) else 0
        features['score_diff'] = score_diff
        features['home_game'] = 1 if home_away == 'H' else 0
        
        # Late inning indicator
        features['late_inning'] = 1 if inning >= 7 else 0
        
        return features

    @staticmethod
    def create_team_features(team_data: Dict) -> pd.DataFrame:
        """
        Create team-level features.
        
        Args:
            team_data: Dictionary with team statistics
            
        Returns:
            DataFrame with team features
        """
        features = pd.DataFrame(index=[0])
        
        features['team_k_rate'] = team_data.get('strikeout_rate', 0)
        features['team_batting_avg'] = team_data.get('batting_avg', 0)
        features['team_obp'] = team_data.get('obp', 0)
        features['team_slugging'] = team_data.get('slugging', 0)
        features['team_runs_per_game'] = team_data.get('runs_per_game', 0)
        features['team_win_pct'] = team_data.get('win_pct', 0)
        
        return features

    @staticmethod
    def combine_features(*feature_dfs: pd.DataFrame) -> pd.DataFrame:
        """
        Combine multiple feature DataFrames.
        
        Args:
            *feature_dfs: Variable number of feature DataFrames
            
        Returns:
            Combined DataFrame
        """
        return pd.concat(feature_dfs, axis=1)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'median', 'mean', or 'forward_fill'
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill').fillna(df.median())
        
        return df

    @staticmethod
    def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional derived features from existing ones.
        
        Args:
            df: Base feature DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Interactions
        if 'k_per_9' in df.columns and 'spin_rate' in df.columns:
            df['k9_spin_interaction'] = df['k_per_9'] * (df['spin_rate'] / 2500)
        
        # Ratios
        if 'strikeouts' in df.columns and 'innings_pitched' in df.columns:
            df['strikeouts_per_inning'] = df['strikeouts'] / (df['innings_pitched'] + 0.1)
        
        # Combinations
        if 'era' in df.columns and 'whip' in df.columns:
            df['era_whip_product'] = df['era'] * df['whip']
        
        return df
