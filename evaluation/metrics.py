"""Metrics calculation for model evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
from pathlib import Path


class MetricsCalculator:
    """Calculate metrics for classification models."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate all metrics for given predictions.
        
        Args:
            y_true: actual labels
            y_pred: predicted labels  
            y_proba: prediction probabilities (optional, not used)
            
        Returns:
            dict of metric scores
        """
        results = {}
        
        # basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # also calculate macro versions for better insight
        # results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        # results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        # results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return results
    
    def get_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Get confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        
        return cm
    
    def get_report(self, y_true, y_pred, class_names=None):
        """Get classification report as dict."""
        return classification_report(y_true, y_pred, 
                                     target_names=class_names,
                                     output_dict=True,
                                     zero_division=0)
    
    def save_metrics(self, metrics, model_name):
        """Save metrics to json file."""
        output_dir = Path(self.config.get('output', {}).get('metrics_dir', 'reports/metrics'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Saved metrics: {filepath}")