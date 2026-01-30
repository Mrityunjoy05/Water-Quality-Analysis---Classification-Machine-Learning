"""Random Forest model."""

from sklearn.ensemble import RandomForestClassifier
import yaml
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self, config):
        super().__init__(config, "Random Forest")
        params = self._load_params()
        self.model = self.build_model(params)
    
    def _load_params(self):
        """Load parameters from config file."""
        try:
            with open('config/model_params.yaml', 'r') as f:
                all_params = yaml.safe_load(f)
            return all_params.get('random_forest', {})
        except:
            return {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def build_model(self, params):
        """Build Random Forest."""
        return RandomForestClassifier(**params)