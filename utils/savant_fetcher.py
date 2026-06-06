"""Utilities to fetch and merge Baseball Savant / Statcast stats.

This module provides helper functions to:
- ingest a pre-downloaded Baseball Savant / PyBaseball DataFrame (recommended)
- merge those advanced metrics into the pitcher feature rows used by the model

Notes:
- Fetching directly from Baseball Savant is best done with the `pybaseball` package (https://github.com/jldbc/pybaseball).
- PyBaseball offers a number of functions for statcast and leaderboards. Example workflows:
    from pybaseball import statcast_pitcher
    df = statcast_pitcher(start_dt, end_dt)

- Because Baseball Savant/Statcast endpoints and pybaseball wrappers change over time, this module separates fetching (left to the user) from merging into features.
"""

import pandas as pd
from typing import Optional


def enrich_features_with_savant(features_df: pd.DataFrame, savant_df: pd.DataFrame, on: str = 'pitcher_id') -> pd.DataFrame:
    """
    Merge Baseball Savant / Statcast-derived advanced metrics into the feature DataFrame.

    Args:
        features_df: DataFrame with one row per pitcher (index 0..n-1) and columns including 'pitcher_id' or other key
        savant_df: DataFrame from pybaseball or downloaded CSV with advanced metrics. Must include same key column.
        on: Column name to join on (default 'pitcher_id'). If not available, you can join on 'pitcher_name' but beware of mismatches.

    Returns:
        DataFrame with additional columns added (where available):
            - csw_pct (Called Strikes + Whiffs as % of total pitches)
            - whiff_pct (Whiff rate on swings)
            - chase_rate (Swings outside zone / Swings total)
            - zone_rate (Percentage of pitches in zone)
            - avg_spin_rate (average spin rate)
            - release_spin_rate etc. (anything present in savant_df will be carried over)

    Example:
        # using pybaseball to fetch statcast / leaderboards first (outside this module):
        from pybaseball import statcast_pitcher
        savant = statcast_pitcher('2026-06-01', '2026-06-06')
        features = enrich_features_with_savant(features, savant, on='pitcher_name')
    """

    if on not in savant_df.columns:
        raise ValueError(f"Join key '{on}' not present in savant_df columns: {savant_df.columns.tolist()}")

    # Select a conservative set of columns that are commonly useful in modeling
    cand_cols = [
        'csw_pct',
        'whiff_pct',
        'chase_rate',
        'zone_rate',
        'avg_spin_rate',
        'spin_rate',
        'avg_release_spin_rate',
        'avg_pitch_speed',
        'release_speed',
        'pfx_x',
        'pfx_z'
    ]

    present = [c for c in cand_cols if c in savant_df.columns]

    cols_to_merge = [on] + present

    merge_df = savant_df[cols_to_merge].drop_duplicates(subset=[on])

    merged = pd.merge(features_df.reset_index(drop=True), merge_df, on=on, how='left')

    # Fill missing advanced metrics with sensible defaults (0 or median)
    for c in present:
        if merged[c].isnull().any():
            median = merged[c].median()
            merged[c] = merged[c].fillna(median if not pd.isna(median) else 0)

    return merged


def example_pybaseball_fetch(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Example helper showing how to fetch statcast/pitching data using pybaseball.

    This function will attempt to import pybaseball and call a reasonable function. If pybaseball is not installed
    or the environment doesn't allow downloads, this will raise an informative error.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'

    Returns:
        DataFrame returned by pybaseball function (developer may need to adapt call depending on pybaseball version)
    """
    try:
        import pybaseball as pb
    except Exception as e:
        raise ImportError("pybaseball is required to fetch Baseball Savant data. Install it with `pip install pybaseball`.")

    # The exact pybaseball function to call depends on the version and what you need.
    # One common call is `statcast_pitcher(start_dt, end_dt)` which returns pitch-level and aggregated pitcher metrics.
    try:
        # statcast_pitcher returns a DataFrame with pitcher-level stats for a period in some pybaseball versions
        df = pb.statcast_pitcher(start_date, end_date)
        return df
    except Exception:
        # Fall back to statcast for all pitches and let the user aggregate externally
        try:
            df = pb.statcast(start_date, end_date)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Statcast data via pybaseball: {e}")
