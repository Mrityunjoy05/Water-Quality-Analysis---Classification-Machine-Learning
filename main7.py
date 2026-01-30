"""Main training pipeline for water quality classification."""

import yaml
from pathlib import Path

# import modules
from core import DataLoader, DataValidator, DataPreprocessor , ImbalanceHandler
from features import FeatureEngineer, FeatureSelector
from training import ModelTrainer, HyperparameterTuner
from evaluation import ModelEvaluator


def load_config():
    """Load configuration from yaml file."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run complete training pipeline."""
    
    print("\n" + "="*60)
    print("WATER QUALITY CLASSIFICATION - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # load config
    config = load_config()
    
    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    
    loader = DataLoader(config)
    df = loader.load_data()
    loader.print_data_summary()
    
    # ==========================================
    # STEP 2: VALIDATE DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 2: DATA VALIDATION")
    print("="*60)
    
    validator = DataValidator()
    validator.validate_dataframe(df)
    
    target_col = config.get('data', {}).get('target_column')
    validator.check_target_distribution(df, target_col)
    
    # ==========================================
    # STEP 3: PREPROCESS DATA
    # ==========================================
    print("\n" + "="*60)
    print("STEP 3: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df, target_col)
    
    print("End of Train")
    feature_names = preprocessor.get_feature_names()
    class_names = preprocessor.get_target_classes()
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Classes: {class_names}")
    
    # save preprocessor
    preprocessor.save_preprocessor()
    # ==========================================
    # STEP 3.5: HANDLE IMBALANCE (New Step)
    # ==========================================
    use_smote = config.get('training', {}).get('use_smote', False)
    
    if use_smote:
        print("\n" + "="*60)
        print("STEP 3.5: CLASS IMBALANCE CORRECTION")
        print("="*60)
        
        handler = ImbalanceHandler(config)
        X_train, y_train = handler.handle_imbalance(X_train, y_train)
    # ==========================================
    # STEP 4: FEATURE ENGINEERING (Optional)
    # ==========================================
    # Uncomment if you want to use feature engineering
    import pandas as pd
    engineer = FeatureEngineer()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_train_df = engineer.create_features(X_train_df)
    X_train = X_train_df.values
    
    # ==========================================
    # STEP 5: FEATURE SELECTION (Optional)
    # ==========================================
    use_feature_selection = config.get('features', {}).get('feature_selection', {}).get('enabled', False)
    
    if use_feature_selection:
        print("\n" + "="*60)
        print("STEP 4: FEATURE SELECTION")
        print("="*60)
        
        selector = FeatureSelector()
        n_features = config.get('features', {}).get('feature_selection', {}).get('n_features', 20)
        
        import pandas as pd
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        selected_features = selector.select_features(X_train_df, y_train, n_features)
        
        # transform both train and test
        X_train = selector.transform(X_train_df).values
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        X_test = selector.transform(X_test_df).values
        
        feature_names = selected_features
        
        # save importance
        selector.save_importance('reports/feature_importance.csv')
    
    # ==========================================
    # STEP 6: TRAIN MODELS
    # ==========================================
    print("\n" + "="*60)
    print("STEP 5: MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer(config)
    models = trainer.train_all_models(X_train, y_train, X_test, y_test, feature_names)
    
    # ==========================================
    # STEP 7: HYPERPARAMETER TUNING (Optional)
    # ==========================================
    tune_models = config.get('training', {}).get('hyperparameter_tuning', {}).get('enabled', False)
    
    if tune_models:
        print("\n" + "="*60)
        print("STEP 6: HYPERPARAMETER TUNING")
        print("="*60)
        
        tuner = HyperparameterTuner(config)
        best_models = tuner.tune_all_models(X_train, y_train)
        
        # replace with tuned models
        models = best_models
    
    # ==========================================
    # STEP 8: EVALUATE MODELS
    # ==========================================
    print("\n" + "="*60)
    print("STEP 7: MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_all(models, X_test, y_test, class_names)
    
    # plot feature importance for each model
    for name, model in models.items():
        if hasattr(model, 'get_feature_importance'):
            evaluator.plot_feature_importance(model, feature_names, name, top_n=15)
    
    # ==========================================
    # DONE
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    
    print("\nOutputs saved to:")
    print("  - Models: saved_models/")
    print("  - Metrics: reports/metrics/")
    print("  - Plots: reports/figures/")
    
    print("\nTo run the web app:")
    print("  streamlit run app.py")
    
    return models, results


if __name__ == "__main__":
    main()