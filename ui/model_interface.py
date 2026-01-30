"""Model interface for predictions."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class ModelInterface:
    """Interface for loading models and making predictions."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_name = None
        self.class_names = None
    
    def load_model(self, model_name, use_tuned=True):
        """Load a trained model.
        
        Args:
            model_name: name of model to load
            use_tuned: whether to use tuned model (default: True)
        
        Returns:
            True if successful
        """
        model_dir = Path('saved_models') / model_name.lower().replace(' ', '_')
        
        if use_tuned:
            model_file = model_dir / 'model_tuned.pkl'
            if not model_file.exists():
                model_file = model_dir / 'model.pkl'
        else:
            model_file = model_dir / 'model.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # load model
        model_data = joblib.load(model_file)
        self.model = model_data['model']
        self.model_name = model_data.get('model_name', model_name)
        
        print(f"Loaded model: {self.model_name}")
        return True
    
    def load_preprocessor(self):
        """Load the preprocessor."""
        preprocessor_file = Path('saved_models/preprocessor.pkl')
        
        if not preprocessor_file.exists():
            raise FileNotFoundError("Preprocessor not found")
        
        preprocessor_data = joblib.load(preprocessor_file)
        self.preprocessor = preprocessor_data
        
        # get class names if available
        if 'target_encoder' in preprocessor_data:
            target_enc = preprocessor_data['target_encoder']
            if hasattr(target_enc, 'classes_'):
                self.class_names = target_enc.classes_.tolist()
        
        print("Loaded preprocessor")
        return True
    
    def preprocess_data(self, df):
        """Preprocess input data using saved preprocessor.
        
        Args:
            df: input dataframe
        
        Returns:
            preprocessed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # apply same transformations
        scaler = self.preprocessor.get('scaler')
        label_encoders = self.preprocessor.get('label_encoders', {})
        feature_names = self.preprocessor.get('feature_names')
        
        # encode categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
        
        # select only features used in training
        if feature_names:
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                # add missing features with 0
                for feat in missing_features:
                    df[feat] = 0
            
            df = df[feature_names]
        
        # scale features
        if scaler:
            X = scaler.transform(df)
        else:
            X = df.values
        
        return X
    
    def predict(self, X):
        """Make predictions.
        
        Args:
            X: preprocessed features
        
        Returns:
            predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities.
        
        Args:
            X: preprocessed features
        
        Returns:
            prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def predict_from_dataframe(self, df):
        """End-to-end prediction from raw dataframe.
        
        Args:
            df: raw input dataframe
        
        Returns:
            dict with predictions and probabilities
        """
        # preprocess
        X = self.preprocess_data(df.copy())
        
        # predict
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # convert to class names if available
        if self.class_names:
            pred_classes = [self.class_names[p] for p in predictions]
        else:
            pred_classes = predictions.tolist()
        
        results = {
            'predictions': pred_classes,
            'prediction_indices': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        return results
    
    def get_available_models(self):
        """Get list of available trained models."""
        models_dir = Path('saved_models')
        
        if not models_dir.exists():
            return []
        
        available = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # check if model files exist
                if (model_dir / 'model.pkl').exists() or (model_dir / 'model_tuned.pkl').exists():
                    # convert dir name to model name
                    name = model_dir.name.replace('_', ' ').title()
                    available.append(name)
        
        return available
    
    def load_metrics(self, model_name):
        """Load saved metrics for a model.
        
        Args:
            model_name: name of model
        
        Returns:
            dict of metrics
        """
        metrics_file = Path('reports/metrics') / f"{model_name.lower().replace(' ', '_')}_metrics.json"
        
        if not metrics_file.exists():
            return None
        
        import json
        with open(metrics_file, 'r') as f:
            return json.load(f)