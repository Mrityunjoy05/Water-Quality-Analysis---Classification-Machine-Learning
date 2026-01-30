"""Data loading functionality for Water Quality Classification."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class DataLoader:
    """Load and manage water quality data."""
    
    def __init__(self, config: dict):
        """Initialize DataLoader.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.data = None
        self.data_info = {}
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load water quality data from CSV file.
        
        Args:
            filepath: Path to CSV file (uses config if not provided)
            
        Returns:
            DataFrame with loaded data
        """
        if filepath is None:
            filepath = self.config.get('data', {}).get('raw_data_path')
        
        if not filepath:
            raise ValueError("Data path not provided in config or arguments")
        
        try:
            # Get encoding from config
            encoding = self.config.get('data', {}).get('encoding', 'utf-8')
            
            print(f"\n{'='*60}")
            print(f"📂 Loading data from: {filepath}")
            print(f"{'='*60}")
            
            self.data = pd.read_csv(filepath, encoding=encoding)
            
            print(f"✅ Successfully loaded!")
            print(f"   Rows: {len(self.data)}")
            print(f"   Columns: {len(self.data.columns)}")
            
            # Store data information
            self._compute_data_info()
            
            return self.data
        
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def _compute_data_info(self) -> None:
        """Compute and store basic data information."""
        if self.data is None:
            return
        
        self.data_info = {
            'n_rows': len(self.data),
            'n_columns': len(self.data.columns),
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'missing_counts': self.data.isnull().sum().to_dict(),
            'missing_percentages': (self.data.isnull().sum() / len(self.data) * 100).round(2).to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
    
    def get_data_info(self) -> Dict:
        """Get basic information about the dataset.
        
        Returns:
            Dictionary with data information
        """
        return self.data_info
    
    def print_data_summary(self) -> None:
        """Print a summary of the loaded data."""
        if self.data is None:
            print("⚠️ No data loaded yet")
            return
        
        print(f"\n{'='*60}")
        print("📊 DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total Rows      : {self.data_info['n_rows']}")
        print(f"Total Columns   : {self.data_info['n_columns']}")
        print(f"Memory Usage    : {self.data_info['memory_usage_mb']:.2f} MB")
        print(f"{'='*60}")
        
        # Show columns with missing values
        missing_cols = {k: v for k, v in self.data_info['missing_percentages'].items() if v > 0}
        
        if missing_cols:
            print(f"\n🔍 Columns with Missing Values ({len(missing_cols)}):")
            print(f"{'-'*60}")
            for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:15]:
                count = self.data_info['missing_counts'][col]
                print(f"  • {col[:40]:<40} : {count:>4} ({pct:>5.2f}%)")
        else:
            print("\n✅ No missing values found")
        
        print(f"{'='*60}\n")
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """Get columns grouped by data type.
        
        Returns:
            Dictionary with column types
        """
        if self.data is None:
            return {}
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\n📋 Column Types:")
        print(f"   Numeric    : {len(numeric_cols)}")
        print(f"   Categorical: {len(categorical_cols)}")
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
    
    def get_target_distribution(self, target_col: Optional[str] = None) -> pd.Series:
        """Get distribution of target variable.
        
        Args:
            target_col: Name of target column (uses config if not provided)
            
        Returns:
            Series with value counts
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if target_col is None:
            target_col = self.config.get('data', {}).get('target_column')
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        distribution = self.data[target_col].value_counts()
        
        print(f"\n🎯 Target Variable: '{target_col}'")
        print(f"{'-'*60}")
        for idx, (label, count) in enumerate(distribution.items(), 1):
            pct = (count / len(self.data)) * 100
            print(f"  {idx}. {str(label):<30} : {count:>4} ({pct:>5.2f}%)")
        print(f"{'-'*60}\n")
        
        return distribution
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Name of output file
        """
        output_path = Path(self.config.get('data', {}).get('processed_data_path', 'data/processed/'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        df.to_csv(filepath, index=False)
        
        print(f"💾 Processed data saved to: {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            DataFrame with processed data
        """
        filepath = Path(self.config.get('data', {}).get('processed_data_path', 'data/processed/')) / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        print(f"📂 Loading processed data from: {filepath}")
        return pd.read_csv(filepath)