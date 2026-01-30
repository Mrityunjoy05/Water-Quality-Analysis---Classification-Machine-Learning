"""Handling class imbalance using SMOTE."""

import pandas as pd
from imblearn.over_sampling import SMOTE

class ImbalanceHandler:
    """Handle class imbalance in the training dataset."""
    
    def __init__(self, config):
        self.config = config
        self.sampling_strategy = config.get('training', {}).get('smote_strategy', 'auto')
        self.random_state = config.get('training', {}).get('random_state', 42)
        self.smote = SMOTE(
            sampling_strategy=self.sampling_strategy, 
            random_state=self.random_state
        )

    def handle_imbalance(self, X_train, y_train):
        """
        Apply SMOTE to balance the classes.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            
        Returns:
            X_resampled, y_resampled
        """
        print("\n HANDLING CLASS IMBALANCE")
        
        # Check distribution before
        before_dist = pd.Series(y_train).value_counts().to_dict()
        print(f"   Distribution before SMOTE: {before_dist}")
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        # Check distribution after
        after_dist = pd.Series(y_resampled).value_counts().to_dict()
        print(f"   Distribution after SMOTE:  {after_dist}")
        print(f"   ✅ Created {len(y_resampled) - len(y_train)} synthetic samples.")
        
        return X_resampled, y_resampled