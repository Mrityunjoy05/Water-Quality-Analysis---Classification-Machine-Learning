"""UI components for Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


class UIComponents:
    """Reusable UI components for Streamlit."""
    
    @staticmethod
    def show_header():
        """Display app header."""
        st.set_page_config(
            page_title="Water Quality Classifier",
            page_icon="💧",
            layout="wide"
        )
        
        st.title("💧 Water Quality Classification System")
        st.markdown("---")
    
    @staticmethod
    def show_sidebar_info():
        """Display sidebar information."""
        st.sidebar.title("About")
        st.sidebar.info(
            """
            This app classifies water quality based on 
            various physical, chemical, and biological parameters.
            
            **Models:**
            - Decision Tree
            - Random Forest
            - XGBoost
            - Logistic Regression
            """
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Navigation")
    
    @staticmethod
    def file_uploader():
        """File upload component."""
        st.subheader("📁 Upload Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload water quality data in CSV format"
        )
        
        return uploaded_file
    
    @staticmethod
    def show_dataframe(df, title="Data Preview"):
        """Display dataframe with title."""
        st.subheader(title)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing", df.isnull().sum().sum())
        
        st.dataframe(df.head(50), use_container_width=True)
    
    @staticmethod
    def model_selector():
        """Model selection dropdown."""
        models = [
            "Decision Tree",
            "Random Forest", 
            "XGBoost",
            "Logistic Regression"
        ]
        
        selected = st.selectbox(
            "Select Model",
            models,
            help="Choose which model to use for predictions"
        )
        
        return selected
    
    @staticmethod
    def show_metrics(metrics, model_name):
        """Display model metrics in cards."""
        st.subheader(f"📊 {model_name} Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    
    @staticmethod
    def show_confusion_matrix(model_name):
        """Display confusion matrix image."""
        figures_dir = Path("reports/figures")
        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        filepath = figures_dir / filename
        
        if filepath.exists():
            st.subheader("Confusion Matrix")
            st.image(str(filepath), use_column_width=True)
        else:
            st.warning("Confusion matrix not found")
    
    @staticmethod
    def show_feature_importance(model_name):
        """Display feature importance plot."""
        figures_dir = Path("reports/figures")
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        filepath = figures_dir / filename
        
        if filepath.exists():
            st.subheader("Feature Importance")
            st.image(str(filepath), use_column_width=True)
        else:
            st.info("Feature importance not available")
    
    @staticmethod
    def show_comparison_chart():
        """Display model comparison chart."""
        filepath = Path("reports/figures/model_comparison.png")
        
        if filepath.exists():
            st.subheader("📈 Model Comparison")
            st.image(str(filepath), use_column_width=True)
        else:
            st.warning("Model comparison chart not found")
    
    @staticmethod
    def show_prediction_result(prediction, probability=None, class_names=None):
        """Display prediction result."""
        st.success("Prediction Complete!")
        
        if class_names and prediction < len(class_names):
            pred_class = class_names[prediction]
        else:
            pred_class = f"Class {prediction}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Class", pred_class)
        
        if probability is not None:
            with col2:
                st.metric("Confidence", f"{probability:.2%}")
            
            # show all class probabilities
            if class_names:
                st.subheader("Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': class_names,
                    'Probability': probability
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True)
    
    @staticmethod
    def download_button(data, filename, label="Download"):
        """Create download button for data."""
        st.download_button(
            label=label,
            data=data,
            file_name=filename,
            mime='text/csv'
        )
    
    @staticmethod
    def show_error(message):
        """Display error message."""
        st.error(f"❌ {message}")
    
    @staticmethod
    def show_success(message):
        """Display success message."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def show_info(message):
        """Display info message."""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def show_warning(message):
        """Display warning message."""
        st.warning(f"⚠️ {message}")