"""XGBoost model."""

from xgboost import XGBClassifier
import yaml
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier."""
    
    def __init__(self, config):
        super().__init__(config, "XGBoost")
        params = self._load_params()
        self.model = self.build_model(params)
    
    def _load_params(self):
        """Load parameters from config file."""
        try:
            with open('config/model_params.yaml', 'r') as f:
                all_params = yaml.safe_load(f)
            return all_params.get('xgboost', {})
        except:
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def build_model(self, params):
        """Build XGBoost."""
        return XGBClassifier(**params)