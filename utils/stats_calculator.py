"""Calculate advanced statistics."""

import pandas as pd
import numpy as np
from typing import Dict, List


class StatsCalculator:
    """Calculate pitcher and matchup statistics."""

    @staticmethod
    def calculate_k_per_9(strikeouts: float, innings_pitched: float) -> float:
        """
        Calculate strikeouts per 9 innings.
        
        Args:
            strikeouts: Total strikeouts
            innings_pitched: Total innings pitched
            
        Returns:
            K/9 rate
        """
        if innings_pitched == 0:
            return 0
        return (strikeouts / innings_pitched) * 9

    @staticmethod
    def calculate_whip(walks: float, hits: float, innings_pitched: float) -> float:
        """
        Calculate WHIP (Walks + Hits per IP).
        
        Args:
            walks: Total walks
            hits: Total hits
            innings_pitched: Total innings pitched
            
        Returns:
            WHIP
        """
        if innings_pitched == 0:
            return 0
        return (walks + hits) / innings_pitched

    @staticmethod
    def calculate_fip(home_runs: float, walks: float, strikeouts: float,
                     hit_batsmen: float, innings_pitched: float) -> float:
        """
        Calculate FIP (Fielding Independent Pitching).
        
        Args:
            home_runs: Home runs allowed
            walks: Walks
            strikeouts: Strikeouts
            hit_batsmen: Hit by pitch
            innings_pitched: Innings pitched
            
        Returns:
            FIP
        """
        if innings_pitched == 0:
            return 0
        
        constant = 3.20  # Empirically derived constant
        fip = ((13 * home_runs + 3 * (walks + hit_batsmen) - 2 * strikeouts) / innings_pitched) + constant
        
        return fip

    @staticmethod
    def calculate_strikeout_pct(strikeouts: float, plate_appearances: float) -> float:
        """
        Calculate strikeout percentage.
        
        Args:
            strikeouts: Total strikeouts
            plate_appearances: Total plate appearances against
            
        Returns:
            Strikeout percentage
        """
        if plate_appearances == 0:
            return 0
        return (strikeouts / plate_appearances) * 100

    @staticmethod
    def calculate_rolling_average(series: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate rolling average.
        
        Args:
            series: Data series
            window: Window size
            
        Returns:
            Rolling average series
        """
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def calculate_pitcher_vs_batter_stats(
        matchup_history: pd.DataFrame,
        pitcher_id: str,
        batter_id: str
    ) -> Dict:
        """
        Calculate pitcher vs batter specific statistics.
        
        Args:
            matchup_history: Historical matchup data
            pitcher_id: Pitcher identifier
            batter_id: Batter identifier
            
        Returns:
            Dictionary with matchup stats
        """
        matchups = matchup_history[
            (matchup_history['pitcher_id'] == pitcher_id) &
            (matchup_history['batter_id'] == batter_id)
        ]
        
        if len(matchups) == 0:
            return {
                'at_bats': 0,
                'strikeout_pct': 0,
                'batting_avg': 0,
                'slugging': 0
            }
        
        stats = {
            'at_bats': len(matchups),
            'strikeout_pct': (matchups['is_strikeout'].sum() / len(matchups)) * 100,
            'batting_avg': 1 - (matchups['is_strikeout'].sum() / len(matchups)),
            'slugging': matchups['total_bases'].sum() / len(matchups),
        }
        
        return stats

    @staticmethod
    def calculate_handedness_splits(
        pitcher_data: pd.DataFrame,
        pitcher_id: str
    ) -> Dict:
        """
        Calculate pitcher performance splits by batter handedness.
        
        Args:
            pitcher_data: Pitcher performance data
            pitcher_id: Pitcher identifier
            
        Returns:
            Dictionary with splits
        """
        pitcher_games = pitcher_data[pitcher_data['pitcher_id'] == pitcher_id]
        
        splits = {}
        
        for hand in ['L', 'R']:
            hand_data = pitcher_games[pitcher_games['batter_handedness'] == hand]
            
            if len(hand_data) > 0:
                splits[f'vs_{hand}'] = {
                    'k_per_9': StatsCalculator.calculate_k_per_9(
                        hand_data['strikeouts'].sum(),
                        hand_data['innings_pitched'].sum()
                    ),
                    'era': hand_data['era'].mean(),
                    'at_bats': len(hand_data),
                }
            else:
                splits[f'vs_{hand}'] = {'k_per_9': 0, 'era': 0, 'at_bats': 0}
        
        return splits

    @staticmethod
    def calculate_consistency_score(recent_stats: List[float], recent_performances: int = 10) -> float:
        """
        Calculate pitcher consistency score.
        
        Args:
            recent_stats: Recent performance values
            recent_performances: Number of recent games to consider
            
        Returns:
            Consistency score (0-100)
        """
        if len(recent_stats) < 2:
            return 50  # Default consistency
        
        recent_stats = recent_stats[-recent_performances:]
        
        mean_stat = np.mean(recent_stats)
        std_stat = np.std(recent_stats)
        
        if mean_stat == 0:
            return 50
        
        # Lower coefficient of variation = higher consistency
        cv = std_stat / mean_stat if mean_stat != 0 else 0
        consistency = 100 * (1 - min(cv, 1))
        
        return max(0, min(100, consistency))

    @staticmethod
    def calculate_trend(recent_values: List[float]) -> str:
        """
        Calculate trend of recent performance.
        
        Args:
            recent_values: Recent performance values
            
        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(recent_values) < 2:
            return 'stable'
        
        recent_values = recent_values[-5:]  # Last 5 games
        
        early_avg = np.mean(recent_values[:2])
        recent_avg = np.mean(recent_values[-2:])
        
        change_pct = ((recent_avg - early_avg) / early_avg) if early_avg != 0 else 0
        
        if change_pct > 0.1:
            return 'improving'
        elif change_pct < -0.1:
            return 'declining'
        else:
            return 'stable'
