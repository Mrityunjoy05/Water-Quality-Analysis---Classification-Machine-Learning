"""Base model class for all ML models."""

from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import time


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, config: dict, model_name: str):
        """Initialize base model.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_time = 0.0
        self.feature_names = None
    
    @abstractmethod
    def build_model(self, params: Dict[str, Any]) -> Any:
        """Build and return the model instance.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: Optional[list] = None) -> 'BaseModel':
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
            
        Returns:
            self
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name}")
        print(f"{'='*60}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train))}")
        
        self.feature_names = feature_names
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        
        print(f"   ✅ Training completed in {self.training_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_name} does not support probability predictions")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances.
        
        Returns:
            Feature importances array or None
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute values of coefficients
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            return None
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save trained model to file.
        
        Args:
            filepath: Path to save model (uses default if not provided)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filepath is None:
            save_dir = Path(self.config.get('output', {}).get('saved_models_dir', 'saved_models'))
            save_dir = save_dir / self.model_name.lower().replace(' ', '_')
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / 'model.pkl'
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_time': self.training_time,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 {self.model_name} saved to: {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """Load trained model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            self
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_time = model_data['training_time']
        self.feature_names = model_data.get('feature_names')
        self.is_trained = model_data['is_trained']
        
        print(f"📂 {self.model_name} loaded from: {filepath}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.model is None:
            return {}
        return self.model.get_params()
    
    def __str__(self) -> str:
        """String representation."""
        status = "Trained" if self.is_trained else "Not Trained"
        return f"{self.model_name} ({status})"
    
    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()