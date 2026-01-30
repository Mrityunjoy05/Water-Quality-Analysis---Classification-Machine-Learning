"""Data validation functionality."""

import pandas as pd
import numpy as np
from typing import List, Dict


class DataValidator:
    """Validate data quality and integrity."""
    
    def __init__(self):
        """Initialize DataValidator."""
        self.validation_report = {}
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Run all validation checks on DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if all validations pass
        """
        print(f"\n{'='*60}")
        print("🔍 STARTING DATA VALIDATION")
        print(f"{'='*60}\n")
        
        validations = [
            self._check_empty_dataframe(df),
            self._check_duplicate_rows(df),
            self._check_missing_values(df),
            self._check_data_types(df),
            self._check_outliers(df)
        ]
        
        all_valid = all(validations)
        
        print(f"\n{'='*60}")
        if all_valid:
            print("✅ ALL VALIDATIONS PASSED")
        else:
            print("⚠️ SOME VALIDATIONS FAILED - Check details above")
        print(f"{'='*60}\n")
        
        return all_valid
    
    def _check_empty_dataframe(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is empty."""
        is_valid = not df.empty
        self.validation_report['empty_check'] = is_valid
        
        if not is_valid:
            print("❌ DataFrame is empty!")
        else:
            print("✅ DataFrame is not empty")
        
        return is_valid
    
    def _check_duplicate_rows(self, df: pd.DataFrame) -> bool:
        """Check for duplicate rows."""
        n_duplicates = df.duplicated().sum()
        self.validation_report['n_duplicates'] = n_duplicates
        
        if n_duplicates > 0:
            pct = n_duplicates / len(df) * 100
            print(f"⚠️ Found {n_duplicates} duplicate rows ({pct:.2f}%)")
            return False
        else:
            print("✅ No duplicate rows found")
            return True
    
    def _check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for missing values."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        cols_with_missing = missing[missing > 0]
        self.validation_report['missing_values'] = cols_with_missing.to_dict()
        
        if len(cols_with_missing) > 0:
            print(f"⚠️ Found missing values in {len(cols_with_missing)} columns:")
            for col, count in list(cols_with_missing.items())[:10]:
                pct = missing_pct[col]
                print(f"   • {col[:35]:<35} : {count:>4} ({pct:>5.2f}%)")
            if len(cols_with_missing) > 10:
                print(f"   ... and {len(cols_with_missing) - 10} more columns")
        else:
            print("✅ No missing values found")
        
        return True  # Missing values don't fail validation, just logged
    
    def _check_data_types(self, df: pd.DataFrame) -> bool:
        """Check data types of columns."""
        dtypes = df.dtypes.value_counts()
        self.validation_report['data_types'] = dtypes.to_dict()
        
        print(f"✅ Data types distribution:")
        for dtype, count in dtypes.items():
            print(f"   • {str(dtype):<15} : {count} columns")
        
        return True
    
    def _check_outliers(self, df: pd.DataFrame) -> bool:
        """Check for outliers in numerical columns using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            n_outliers = len(outliers)
            
            if n_outliers > 0:
                outliers_info[col] = {
                    'count': n_outliers,
                    'percentage': round(n_outliers / len(df) * 100, 2)
                }
        
        self.validation_report['outliers'] = outliers_info
        
        if outliers_info:
            print(f"⚠️ Found outliers in {len(outliers_info)} columns:")
            for col, info in list(outliers_info.items())[:10]:
                print(f"   • {col[:35]:<35} : {info['count']:>4} ({info['percentage']:>5.2f}%)")
            if len(outliers_info) > 10:
                print(f"   ... and {len(outliers_info) - 10} more columns")
        else:
            print("✅ No outliers detected")
        
        return True
    
    def check_target_distribution(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Check target variable distribution.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Dictionary with distribution information
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        distribution = df[target_col].value_counts().to_dict()
        distribution_pct = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
        
        print(f"\n🎯 Target Variable Distribution: '{target_col}'")
        print(f"{'-'*60}")
        for class_label, count in distribution.items():
            pct = distribution_pct[class_label]
            print(f"  • Class {class_label:<20} : {count:>4} ({pct:>5.2f}%)")
        
        # Check for class imbalance
        min_class_pct = min(distribution_pct.values())
        max_class_pct = max(distribution_pct.values())
        imbalance_ratio = max_class_pct / min_class_pct
        
        if imbalance_ratio > 2:
            print(f"\n⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        else:
            print("\n✅ Classes are relatively balanced")
        
        print(f"{'-'*60}\n")
        
        return {
            'distribution': distribution,
            'distribution_pct': distribution_pct,
            'n_classes': len(distribution),
            'is_imbalanced': imbalance_ratio > 2,
            'imbalance_ratio': round(imbalance_ratio, 2)
        }
    
    def get_validation_report(self) -> Dict:
        """Get validation report.
        
        Returns:
            Dictionary with validation results
        """
        return self.validation_report