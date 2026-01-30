"""Feature engineering - create new useful features."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create new features from existing ones."""
    
    def __init__(self):
        pass
    
    def create_features(self, df):
        """Create 5-6 useful features for water quality prediction.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with new features
        """
        print(f"\n🔧 Creating new features...")
        df = df.copy()
        created = []
        
        # Feature 1: BOD to COD ratio (pollution indicator)
        if 'bod' in df.columns and 'cod' in df.columns:
            df['bod_cod_ratio'] = df['bod'] / (df['cod'] + 0.01)
            created.append('bod_cod_ratio')
        
        # Feature 2: Dissolved oxygen quality (higher is better)
        if 'dissolved_o2' in df.columns:
            df['do_quality'] = pd.to_numeric(df['dissolved_o2'], errors='coerce')
            df['do_quality'] = df['do_quality'].fillna(df['do_quality'].median())
            created.append('do_quality')
        
        # Feature 3: pH deviation from neutral (7.0)
        if 'ph' in df.columns:
            df['ph_deviation'] = abs(df['ph'] - 7.0)
            created.append('ph_deviation')
        
        # Feature 4: Total dissolved matter (TDS + TSS)
        if 'total_dissolved_solids' in df.columns and 'total_suspended_solids' in df.columns:
            tss = pd.to_numeric(df['total_suspended_solids'], errors='coerce').fillna(0)
            df['total_matter'] = df['total_dissolved_solids'] + tss
            created.append('total_matter')
        
        # Feature 5: Hardness category (soft/hard water indicator)
        if 'hardness_caco3' in df.columns:
            df['is_hard_water'] = (df['hardness_caco3'] > 150).astype(int)
            created.append('is_hard_water')
        
        # Feature 6: Temperature deviation from normal
        if 'temperature' in df.columns:
            df['temp_deviation'] = abs(df['temperature'] - 25.0)
            created.append('temp_deviation')
        
        print(f"   ✅ Created {len(created)} features: {created}")
        return df
    
    def create_interaction_feature(self, df, col1, col2, operation='multiply'):
        """Create interaction between two features."""
        df = df.copy()
        
        if col1 not in df.columns or col2 not in df.columns:
            return df
        
        feature_name = f"{col1}_{operation}_{col2}"
        
        if operation == 'multiply':
            df[feature_name] = df[col1] * df[col2]
        elif operation == 'divide':
            df[feature_name] = df[col1] / (df[col2] + 0.01)
        elif operation == 'add':
            df[feature_name] = df[col1] + df[col2]
        elif operation == 'subtract':
            df[feature_name] = df[col1] - df[col2]
        
        return df
    
    def create_log_features(self, df, columns):
        """Create log-transformed features."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        return df