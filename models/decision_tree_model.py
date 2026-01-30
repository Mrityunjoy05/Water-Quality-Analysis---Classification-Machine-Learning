"""Decision Tree classification model."""

from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any
import yaml
from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree Classifier."""
    
    def __init__(self, config: dict):
        """Initialize Decision Tree model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "Decision Tree")
        
        # Load model parameters
        params = self._load_params()
        self.model = self.build_model(params)
    
    def _load_params(self) -> Dict[str, Any]:
        """Load model parameters from config file."""
        try:
            with open('config/model_params.yaml', 'r') as f:
                all_params = yaml.safe_load(f)
            return all_params.get('decision_tree', {})
        except Exception as e:
            print(f"⚠️ Could not load parameters from config: {e}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            'criterion': 'gini',
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    def build_model(self, params: Dict[str, Any]) -> DecisionTreeClassifier:
        """Build Decision Tree model.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            DecisionTreeClassifier instance
        """
        print(f"🔧 Building {self.model_name} with parameters:")
        for key, value in params.items():
            print(f"   • {key}: {value}")
        
        return DecisionTreeClassifier(**params)