"""Data processing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class DataProcessor:
    """Handle data loading, cleaning, and transformation."""

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """
        Load CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath)

    @staticmethod
    def clean_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean pitcher statistics data.
        
        Args:
            df: Raw pitcher data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove rows with missing pitcher_id
        df = df.dropna(subset=['pitcher_id'])
        
        # Fill missing numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicates based on pitcher_id and game date
        df = df.drop_duplicates(subset=['pitcher_id', 'game_date'], keep='first')
        
        return df

    @staticmethod
    def clean_results_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean results data.
        
        Args:
            df: Raw results data
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Ensure actual_strikeouts is numeric
        df['actual_strikeouts'] = pd.to_numeric(df['actual_strikeouts'], errors='coerce')
        
        # Remove rows with missing strikeouts
        df = df.dropna(subset=['actual_strikeouts'])
        
        # Ensure strikeouts are non-negative integers
        df['actual_strikeouts'] = df['actual_strikeouts'].astype(int)
        df = df[df['actual_strikeouts'] >= 0]
        
        return df

    @staticmethod
    def validate_lineup(lineup: pd.DataFrame) -> bool:
        """
        Validate lineup data.
        
        Args:
            lineup: Lineup DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['pitcher_id', 'pitcher_name', 'team']
        
        for col in required_cols:
            if col not in lineup.columns:
                print(f"Missing required column: {col}")
                return False
        
        if len(lineup) == 0:
            print("Lineup is empty")
            return False
        
        return True

    @staticmethod
    def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize feature values.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Normalized DataFrame
        """
        X = X.copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_min = X[col].min()
            col_max = X[col].max()
            
            if col_max != col_min:
                X[col] = (X[col] - col_min) / (col_max - col_min)
        
        return X

    @staticmethod
    def handle_outliers(X: pd.DataFrame, strategy: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in features.
        
        Args:
            X: Feature DataFrame
            strategy: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        X = X.copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if strategy == 'iqr':
            for col in numeric_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif strategy == 'zscore':
            for col in numeric_cols:
                mean = X[col].mean()
                std = X[col].std()
                
                if std > 0:
                    X[col] = X[col].clip(
                        lower=mean - threshold * std,
                        upper=mean + threshold * std
                    )
        
        return X

    @staticmethod
    def create_time_features(df: pd.DataFrame, date_col: str = 'game_date') -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['quarter'] = df[date_col].dt.quarter
        
        return df

    @staticmethod
    def aggregate_pitcher_stats(df: pd.DataFrame, agg_period: str = 'season') -> pd.DataFrame:
        """
        Aggregate pitcher statistics over period.
        
        Args:
            df: Pitcher data with dates
            agg_period: 'season', 'month', or 'week'
            
        Returns:
            Aggregated statistics
        """
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        if agg_period == 'season':
            grouped = df.groupby('pitcher_id')
        elif agg_period == 'month':
            df['year_month'] = df['game_date'].dt.to_period('M')
            grouped = df.groupby(['pitcher_id', 'year_month'])
        elif agg_period == 'week':
            df['year_week'] = df['game_date'].dt.to_period('W')
            grouped = df.groupby(['pitcher_id', 'year_week'])
        
        agg_stats = grouped.agg({
            'strikeouts': 'sum',
            'innings_pitched': 'sum',
            'k_per_9': 'mean',
            'era': 'mean',
            'whip': 'mean'
        })
        
        return agg_stats
