"""Hyperparameter tuning using Grid Search CV."""

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import yaml
import joblib
from pathlib import Path
from tqdm import tqdm


class HyperparameterTuner:
    """Tune hyperparameters for all models using Grid Search CV."""
    
    def __init__(self, config):
        self.config = config
        self.best_models = {}
        self.best_params = {}
        self.cv_results = {}
    
    def _load_param_grids(self):
        """Load parameter grids from config."""
        try:
            with open('config/model_params.yaml', 'r') as f:
                params = yaml.safe_load(f)
            return params.get('hyperparameter_grids', {})
        except:
            return self._get_default_grids()
    
    def _get_default_grids(self):
        """Default parameter grids if config not found."""
        return {
            'decision_tree': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            }
        }
    
    def tune_all_models(self, X_train, y_train):
        """Tune all models using Grid Search CV.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Dictionary of best models
        """
        print(f"\n{'='*60}")
        print("🔍 HYPERPARAMETER TUNING (Grid Search CV)")
        print(f"{'='*60}\n")
        
        param_grids = self._load_param_grids()
        cv_folds = self.config.get('training', {}).get('cv_folds', 5)
        
        # Model definitions
        models = {
            'Decision Tree': (DecisionTreeClassifier(random_state=42), 
                             param_grids.get('decision_tree', {})),
            'Random Forest': (RandomForestClassifier(random_state=42, n_jobs=-1), 
                             param_grids.get('random_forest', {})),
            'XGBoost': (XGBClassifier(random_state=42, n_jobs=-1), 
                       param_grids.get('xgboost', {})),
            'Logistic Regression': (LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000), 
                                   param_grids.get('logistic_regression', {}))
        }
        
        # Tune each model
        for name, (model, param_grid) in tqdm(models.items(), desc="Tuning models"):
            print(f"\n{'-'*60}")
            print(f"Tuning: {name}")
            print(f"{'-'*60}")
            print(f"Parameter grid: {param_grid}")
            print(f"CV folds: {cv_folds}")
            
            # Grid Search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.best_models[name] = grid_search.best_estimator_
            self.best_params[name] = grid_search.best_params_
            self.cv_results[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_
            }
            
            print(f"\n   Best CV Score: {grid_search.best_score_:.4f}")
            print(f"   Best Params: {grid_search.best_params_}")
            
            # Save tuned model
            self._save_tuned_model(name, grid_search.best_estimator_)
        
        print(f"\n{'='*60}")
        print("✅ ALL MODELS TUNED")
        print(f"{'='*60}\n")
        
        self._print_summary()
        
        return self.best_models
    
    def _save_tuned_model(self, model_name, model):
        """Save tuned model."""
        save_dir = Path(self.config.get('output', {}).get('saved_models_dir', 'saved_models'))
        save_dir = save_dir / model_name.lower().replace(' ', '_')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / 'model_tuned.pkl'
        
        model_data = {
            'model': model,
            'model_name': model_name,
            'best_params': self.best_params[model_name],
            'cv_score': self.cv_results[model_name]['best_score']
        }
        
        joblib.dump(model_data, filepath)
        print(f"   💾 Saved tuned model: {filepath}")
    
    def _print_summary(self):
        """Print tuning summary."""
        print("\n📊 TUNING SUMMARY")
        print(f"{'-'*60}")
        print(f"{'Model':<25} {'Best CV Score'}")
        print(f"{'-'*60}")
        
        for name, results in self.cv_results.items():
            print(f"{name:<25} {results['best_score']:.4f}")
        
        print(f"{'-'*60}\n")
        
        # Best model
        best_model = max(self.cv_results.items(), key=lambda x: x[1]['best_score'])
        print(f"🏆 Best Model: {best_model[0]} (CV Score: {best_model[1]['best_score']:.4f})\n")
        
        # Print best params for each model
        print("\n🔧 BEST PARAMETERS")
        print(f"{'-'*60}")
        for name, params in self.best_params.items():
            print(f"\n{name}:")
            for param, value in params.items():
                print(f"   • {param}: {value}")
    
    def get_best_model(self, model_name):
        """Get best model for a specific classifier."""
        return self.best_models.get(model_name)
    
    def get_best_params(self, model_name):
        """Get best parameters for a specific model."""
        return self.best_params.get(model_name)
    
    def get_results(self):
        """Get all tuning results."""
        return self.cv_results