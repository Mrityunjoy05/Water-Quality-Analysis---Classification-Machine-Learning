"""Feature selection using XGBoost."""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier


class FeatureSelector:
    """Select top N features using XGBoost importance."""
    
    def __init__(self):
        self.model = None
        self.selected_features = None
        self.feature_importance = None
    
    def select_features(self, X, y, n_features=20):
        """Select top N important features using XGBoost.
        
        Args:
            X: Feature dataframe
            y: Target variable
            n_features: Number of top features to select
        
        Returns:
            Selected feature names
        """
        print(f"\n🎯 Selecting top {n_features} features using XGBoost...")
        
        # Train XGBoost
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"   Training XGBoost on {X.shape[1]} features...")
        self.model.fit(X, y)
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Sort by importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top N
        self.selected_features = self.feature_importance.head(n_features)['feature'].tolist()
        
        print(f"   ✅ Selected {len(self.selected_features)} features\n")
        print(f"   Top 10 features:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"      {row['feature'][:35]:<35} : {row['importance']:.4f}")
        
        return self.selected_features
    
    def transform(self, X):
        """Keep only selected features."""
        if self.selected_features is None:
            raise ValueError("No features selected. Run select_features() first.")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            # If numpy array, get indices
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
            return X[:, indices]
    
    def get_importance(self):
        """Get feature importance dataframe."""
        return self.feature_importance
    
    def save_importance(self, filepath):
        """Save feature importance to CSV."""
        if self.feature_importance is not None:
            self.feature_importance.to_csv(filepath, index=False)
            print(f"💾 Feature importance saved: {filepath}")