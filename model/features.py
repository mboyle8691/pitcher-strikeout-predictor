@@
     features = pd.DataFrame()
     
     # Basic strikeout metrics
     features['k_per_9'] = pitcher_data.get('k_per_9', 0)
+    # Advanced Statcast metrics (if provided in pitcher_data dict-like)
+    # Common Baseball Savant metrics to include: csw_pct, whiff_pct, chase_rate, zone_rate, avg_spin_rate
+    features['csw_pct'] = pitcher_data.get('csw_pct', pitcher_data.get('csw', 0))
+    features['whiff_pct'] = pitcher_data.get('whiff_pct', pitcher_data.get('whiff', 0))
+    features['chase_rate'] = pitcher_data.get('chase_rate', pitcher_data.get('o_swing_pct', 0))
+    features['zone_rate'] = pitcher_data.get('zone_rate', pitcher_data.get('zone_pct', 0))
+    features['avg_spin_rate'] = pitcher_data.get('avg_spin_rate', pitcher_data.get('spin_rate', 0))
@@
         features['games_appeared'] = pitcher_data.get('games_appeared', 0)
         
         return features
