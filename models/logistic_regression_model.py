"""Logistic Regression model."""

from sklearn.linear_model import LogisticRegression
import yaml
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self, config):
        super().__init__(config, "Logistic Regression")
        params = self._load_params()
        self.model = self.build_model(params)
    
    def _load_params(self):
        """Load parameters from config file."""
        try:
            with open('config/model_params.yaml', 'r') as f:
                all_params = yaml.safe_load(f)
            return all_params.get('logistic_regression', {})
        except:
            return {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def build_model(self, params):
        """Build Logistic Regression."""
        return LogisticRegression(**params)