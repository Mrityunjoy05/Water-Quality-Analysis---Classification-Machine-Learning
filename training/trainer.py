"""Train all models normally (without hyperparameter tuning)."""

from models import DecisionTreeModel, RandomForestModel, XGBoostModel, LogisticRegressionModel
from tqdm import tqdm


class ModelTrainer:
    """Train all classification models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names=None):
        """Train all 4 models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
        
        Returns:
            Dictionary of trained models
        """
        print(f"\n{'='*60}")
        print("🚀 TRAINING ALL MODELS")
        print(f"{'='*60}\n")
        
        # Initialize all models
        model_classes = {
            'Decision Tree': DecisionTreeModel,
            'Random Forest': RandomForestModel,
            'XGBoost': XGBoostModel,
            'Logistic Regression': LogisticRegressionModel
        }
        
        # Train each model
        for name, ModelClass in tqdm(model_classes.items(), desc="Training models"):
            print(f"\n{'-'*60}")
            print(f"Training: {name}")
            print(f"{'-'*60}")
            
            # Create and train model
            model = ModelClass(self.config)
            model.train(X_train, y_train, feature_names)
            
            # Save model
            model.save_model()
            
            # Store model
            self.models[name] = model
            
            # Quick evaluation
            train_score = model.model.score(X_train, y_train)
            test_score = model.model.score(X_test, y_test)
            
            self.results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'training_time': model.training_time
            }
            
            print(f"\n   Train Accuracy: {train_score:.4f}")
            print(f"   Test Accuracy:  {test_score:.4f}")
            print(f"   Training Time:  {model.training_time:.2f}s")
        
        print(f"\n{'='*60}")
        print("✅ ALL MODELS TRAINED")
        print(f"{'='*60}\n")
        
        self._print_summary()
        
        return self.models
    
    def _print_summary(self):
        """Print training summary."""
        print("\n📊 TRAINING SUMMARY")
        print(f"{'-'*60}")
        print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)'}")
        print(f"{'-'*60}")
        
        for name, results in self.results.items():
            print(f"{name:<25} "
                  f"{results['train_accuracy']:.4f}       "
                  f"{results['test_accuracy']:.4f}       "
                  f"{results['training_time']:.2f}")
        
        print(f"{'-'*60}\n")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"🏆 Best Model: {best_model[0]} (Test Acc: {best_model[1]['test_accuracy']:.4f})\n")
    
    def get_model(self, model_name):
        """Get a specific trained model."""
        return self.models.get(model_name)
    
    def get_results(self):
        """Get training results."""
        return self.results