"""Model evaluation with plots."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .metrics import MetricsCalculator


class ModelEvaluator:
    """Evaluate models and make plots."""
    
    def __init__(self, config):
        self.config = config
        self.metrics_calc = MetricsCalculator(config)
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name, class_names=None):
        """Evaluate a single model.
        
        Args:
            model: trained model
            X_test: test features
            y_test: test labels
            model_name: name of model
            class_names: list of class names (optional)
        
        Returns:
            dict with evaluation results
        """
        print(f"\nEvaluating {model_name}...")
        
        # get predictions
        y_pred = model.predict(X_test)
        
        # calculate metrics
        metrics = self.metrics_calc.calculate_metrics(y_test, y_pred)
        
        # print metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (weighted): {metrics['precision']:.4f}")
        print(f"  Recall (weighted): {metrics['recall']:.4f}")
        print(f"  F1 Score (weighted): {metrics['f1_score']:.4f}")
        
        # get confusion matrix
        cm = self.metrics_calc.get_confusion_matrix(y_test, y_pred)
        cm_pct = self.metrics_calc.get_confusion_matrix(y_test, y_pred, normalize=True)
        
        # save metrics
        self.metrics_calc.save_metrics(metrics, model_name)
        
        # make confusion matrix plot
        self._plot_confusion_matrix(cm, cm_pct, model_name, class_names)
        
        # store results
        results = {
            'metrics': metrics,
            'cm': cm.tolist(),
            'cm_normalized': cm_pct.tolist()
        }
        self.results[model_name] = results
        
        return results
    
    def evaluate_all(self, models, X_test, y_test, class_names=None):
        """Evaluate multiple models."""
        print("\n" + "="*60)
        print("Evaluating all models...")
        print("="*60)
        
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name, class_names)
        
        # make comparison plot
        self._plot_comparison()
        
        print("\nDone! Check reports/ folder for outputs")
        return self.results
    
    def _plot_confusion_matrix(self, cm, cm_pct, model_name, class_names):
        """Plot confusion matrix side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        # counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'{model_name} - Counts')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # percentages
        sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title(f'{model_name} - Percentages')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        
        plt.tight_layout()
        
        # save
        output_dir = Path(self.config.get('output', {}).get('figures_dir', 'reports/figures'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_comparison(self):
        """Plot model comparison."""
        if not self.results:
            return
        
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = [self.results[m]['metrics'].get(metric, 0) for m in models]
            
            ax.bar(models, values)
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1.0])
            ax.tick_params(axis='x', rotation=45)
            
            # add values on bars
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_dir = Path(self.config.get('output', {}).get('figures_dir', 'reports/figures'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        """Plot feature importance for tree models."""
        importance = model.get_feature_importance()
        
        if importance is None:
            print(f"  {model_name} doesn't have feature importance")
            return
        
        # get top features
        indices = np.argsort(importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        # plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # save
        output_dir = Path(self.config.get('output', {}).get('figures_dir', 'reports/figures'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()