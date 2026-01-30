"""Data preprocessing for water quality classification."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict
import joblib
from pathlib import Path


class TargetEncoder:
    """Target encoder for categorical variables."""
    
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.mapping = {}
        self.global_mean = None
    
    def fit(self, X: pd.Series, y: pd.Series):
        self.global_mean = y.mean()
        
        stats = pd.DataFrame({'cat': X, 'target': y})
        agg = stats.groupby('cat')['target'].agg(['mean', 'count'])
        
        smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - self.min_samples_leaf) / self.smoothing))
        self.mapping = (agg['mean'] * smoothing_factor + self.global_mean * (1 - smoothing_factor)).to_dict()
        
        return self
    
    def transform(self, X: pd.Series):
        return X.map(self.mapping).fillna(self.global_mean)


class DataPreprocessor:
    """Handles all data preprocessing steps."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.label_encoders = {}
        self.target_encoders = {}
        self.feature_names = None
        self.target_encoder = None
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names - remove spaces and special characters."""
        print(f"\n🧹 Cleaning column names...")
        
        df = df.copy()
        df.columns = [
            col.strip()
               .replace(" ", "_")
               .replace("(", "")
               .replace(")", "")
               .replace("-", "_")
            for col in df.columns
        ]
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"   ✅ Column names cleaned")
        return df
    
    def remove_invalid_target_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Remove rows with invalid target values."""
        print(f"\n🗑️ Removing invalid target values...")
        
        initial_rows = len(df)
        df = df[df[target_col] != "No Information"].copy()
        removed = initial_rows - len(df)
        
        if removed > 0:
            print(f"   Removed {removed} rows with 'No Information'")
        print(f"   Remaining rows: {len(df)}")
        
        return df
    
    def create_missing_indicators(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create binary indicators for missing values before imputation."""
        print(f"\n🏷️ Creating missing value indicators...")
        
        df = df.copy()
        created = 0
        
        for col in columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                indicator_name = f"{col}_missing"
                df[indicator_name] = df[col].isnull().astype(int)
                created += 1
        
        print(f"   ✅ Created {created} missing indicators")
        return df
    
    # def drop_high_missing_columns(self, df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    #     """Drop columns with missing values above threshold."""
    #     print(f"\n🗑️ Dropping columns with >{threshold}% missing values...")
        
    #     null_percentage = (df.isnull().sum() / len(df)) * 100
    #     columns_to_drop = null_percentage[null_percentage >= threshold]
        
    #     if len(columns_to_drop) > 0:
    #         print(f"   Columns to drop ({len(columns_to_drop)}):")
    #         for col, pct in columns_to_drop.items():
    #             print(f"      • {col[:40]:<40} : {pct:.2f}%")
    #         df = df.drop(columns=columns_to_drop.index)
    #     else:
    #         print(f"   ✅ No columns exceed threshold")
        
    #     return df
    
    
    def drop_high_missing_columns(self, df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
        """Drop specific high-missing columns manually."""
        
        # The two columns you identified for removal
        cols_to_remove = ['use_of_water_in_down_stream', 'remark']
        
        # print(f"\n🗑️ Manually dropping high-missing columns...")
        
        # Check which of these actually exist in the dataframe before dropping
        existing_cols = [col for col in cols_to_remove if col in df.columns]
        
        if existing_cols:
            print(f"   Removing ({len(existing_cols)}): {existing_cols}")
            df = df.drop(columns=existing_cols)
        else:
            print("   ✅ Targeted columns already removed or not found.")
            
        return df

    def impute_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Impute missing values - numeric with median, categorical with 'Unknown'."""
        print(f"\n🔧 Imputing missing values...")
        
        df = df.copy()
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # df[col].fillna(df[col].median(), inplace=True)
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [c for c in categorical_cols if c != target_col]
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # df[col].fillna("Unknown", inplace=True)
                df[col] = df[col].fillna("Unknown")
        
        print(f"   ✅ Imputed {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        return df
    
    def engineer_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date columns."""
        print(f"\n📅 Engineering date features...")
        
        df = df.copy()
        
        # Convert date columns
        if 'sampling_date' in df.columns:
            df['sampling_date'] = pd.to_datetime(df['sampling_date'], format='%d-%m-%Y', errors='coerce')
        
        if 'month' in df.columns:
            df['month'] = pd.to_datetime(df['month'], format='%b', errors='coerce').dt.month
        
        if 'sampling_time' in df.columns:
            df['sampling_time'] = pd.to_datetime(df['sampling_time'], format='%H:%M:%S', errors='coerce').dt.time
        
        # Create combined datetime
        if 'sampling_date' in df.columns and 'sampling_time' in df.columns:
            df['sampling_datetime'] = pd.to_datetime(
                df['sampling_date'].astype(str) + ' ' + df['sampling_time'].astype(str),
                errors='coerce'
            )
        
        # Extract features from datetime columns
        datetime_cols = df.select_dtypes(include='datetime64[ns]').columns.to_list()
        
        for col in datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month_name'] = df[col].dt.month_name()
        
        # Drop original datetime columns
        if datetime_cols:
            df = df.drop(columns=datetime_cols)
            print(f"   ✅ Extracted features from {len(datetime_cols)} datetime columns")
        
        return df
    
    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with only one unique value."""
        print(f"\n🗑️ Removing constant columns...")
        
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        
        if constant_cols:
            print(f"   Constant columns ({len(constant_cols)}): {constant_cols}")
            df = df.drop(columns=constant_cols)
        else:
            print(f"   ✅ No constant columns found")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            print(f"\n🗑️ Removed {n_removed} duplicate rows")
        
        return df
    
    def drop_non_predictive_columns(self, df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
        """Drop non-predictive columns specified in config."""
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        
        if existing_cols:
            print(f"\n🗑️ Dropping {len(existing_cols)} non-predictive columns")
            df = df.drop(columns=existing_cols)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str],
                                   y: Optional[pd.Series] = None,
                                   method: str = 'label') -> pd.DataFrame:
        """Encode categorical features."""
        print(f"\n🔤 Encoding categorical features ({method})...")
        
        df = df.copy()
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f"   ✅ One-hot encoded {len(categorical_cols)} columns")
        
        elif method == 'label':
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            print(f"   ✅ Label encoded {len(categorical_cols)} columns")
        
        elif method == 'target':
            if y is None:
                raise ValueError("Target variable required for target encoding")
            
            smoothing = self.config.get('features', {}).get('target_encoding', {}).get('smoothing', 1.0)
            min_samples = self.config.get('features', {}).get('target_encoding', {}).get('min_samples_leaf', 1)
            
            for col in categorical_cols:
                if col not in self.target_encoders:
                    self.target_encoders[col] = TargetEncoder(smoothing=smoothing, min_samples_leaf=min_samples)
                    df[col] = self.target_encoders[col].fit(df[col], y).transform(df[col])
                else:
                    df[col] = self.target_encoders[col].transform(df[col])
            print(f"   ✅ Target encoded {len(categorical_cols)} columns")
        
        return df
    
    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable."""
        if self.target_encoder is None:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            print(f"\n🎯 Target classes: {list(self.target_encoder.classes_)}")
        else:
            y_encoded = self.target_encoder.transform(y)
        
        return y_encoded
    
    def scale_features(self, X: pd.DataFrame, scaler_type: str = 'standard', fit: bool = True) -> np.ndarray:
        """Scale numerical features."""
        if fit:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            print(f"\n📏 Scaled features using {scaler_type} scaler")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple:
        """Split data into train and test sets."""
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"\n✂️ Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str) -> Tuple:
        """Run complete preprocessing pipeline."""
        print(f"\n{'='*60}")
        print("⚙️ PREPROCESSING PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Clean column names
        df = self.clean_column_names(df)
        
        # 🔍 DEBUG CHECK: See what happened to your columns
        # --- Step 1.5: Sync the target variable ---
        # We must transform the target_col string to match the cleaned DataFrame
        old_target = target_col
        target_col = target_col.strip().replace(" ", "_").lower()
    
        if target_col not in df.columns:
            print(f"❌ ERROR: Target '{target_col}' not found in cleaned columns!")
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Target column '{target_col}' is missing after cleaning.")
        else:
            print(f"✅ Target synchronized: '{old_target}' -> '{target_col}'")

        print(f"🎯 Updated target column name to: '{target_col}'")
        # Step 2: Remove invalid target values
        df = self.remove_invalid_target_values(df, target_col)
        
        # Step 3: Create missing indicators for important columns
        important_cols = ['major_polluting_sources', 'visibility_effluent_discharge']
        df = self.create_missing_indicators(df, important_cols)
        
        # Step 4: Fill missing in those columns
        if 'major_polluting_sources' in df.columns:
            df['major_polluting_sources'].fillna('Unknown', inplace=True)
        if 'visibility_effluent_discharge' in df.columns:
            df['visibility_effluent_discharge'].fillna('Unknown', inplace=True)
        
        # Step 5: Drop high missing columns
        missing_threshold = self.config.get('features', {}).get('missing_threshold', 40.0)
        df = self.drop_high_missing_columns(df, threshold=missing_threshold)
        
        # Step 6: Impute remaining missing values
        df = self.impute_missing_values(df, target_col)
        
        # Step 7: Engineer date features
        df = self.engineer_date_features(df)
        
        # Step 8: Remove constant columns
        df = self.remove_constant_columns(df)
        
        # Step 9: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 10: Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Step 11: Encode target
        y_encoded = self.encode_target(y)
        
        # Step 12: Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        encoding_method = self.config.get('features', {}).get('categorical_encoding', 'label')
        
        if categorical_cols:
            print(f"\n🔍 Found {len(categorical_cols)} categorical columns")
            X = self.encode_categorical_features(X, categorical_cols, y=y_encoded, method=encoding_method)
        
        # Step 13: Split data
        test_size = self.config.get('data', {}).get('train_test_split', 0.2)
        random_state = self.config.get('random_state', 42)
        stratify = self.config.get('data', {}).get('stratify', True)
        
        X_train, X_test, y_train, y_test = self.split_data(
            X, y_encoded, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Step 14: Store feature names
        self.feature_names = X.columns.tolist()
        print(f"\n📊 Final features: {len(self.feature_names)}")
        
        # Step 15: Scale features
        scaling_method = self.config.get('features', {}).get('scaling_method', 'standard')
        X_train_scaled = self.scale_features(X_train, scaler_type=scaling_method, fit=True)
        X_test_scaled = self.scale_features(X_test, scaler_type=scaling_method, fit=False)
        
        print(f"\n{'='*60}")
        print("✅ PREPROCESSING COMPLETE")
        print(f"{'='*60}\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names if self.feature_names else []
    
    def get_target_classes(self) -> np.ndarray:
        if self.target_encoder:
            return self.target_encoder.classes_
        return None
    
    def save_preprocessor(self, filepath: str = "saved_models/preprocessor.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoders': self.target_encoders,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"💾 Preprocessor saved: {filepath}")
    
    def load_preprocessor(self, filepath: str = "saved_models/preprocessor.pkl"):
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.target_encoders = preprocessor_data.get('target_encoders', {})
        self.target_encoder = preprocessor_data['target_encoder']
        self.feature_names = preprocessor_data['feature_names']
        
        print(f"📂 Preprocessor loaded: {filepath}")