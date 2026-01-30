"""Streamlit web application for water quality classification."""

import streamlit as st
import pandas as pd
from ui.components import UIComponents
from ui.model_interface import ModelInterface

# initialize components
ui = UIComponents()
model_interface = ModelInterface()

# show header
ui.show_header()

# sidebar
ui.show_sidebar_info()

# main navigation
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Make Predictions", "Model Evaluation", "About"]
)

# ============================================
# HOME PAGE
# ============================================
if page == "Home":
    st.header("Welcome to Water Quality Classification System")
    
    st.markdown("""
    This application helps classify water quality based on various parameters 
    including physical, chemical, and biological measurements.
    
    ### Features:
    - 🔍 **Make Predictions**: Upload water quality data and get instant classifications
    - 📊 **Model Evaluation**: View detailed performance metrics and visualizations
    - 🤖 **Multiple Models**: Choose from 4 different classification algorithms
    
    ### Available Models:
    1. **Decision Tree** - Simple, interpretable model
    2. **Random Forest** - Ensemble of decision trees for better accuracy
    3. **XGBoost** - Gradient boosting for high performance
    4. **Logistic Regression** - Linear baseline model
    
    ### Getting Started:
    1. Navigate to **Make Predictions** to classify water samples
    2. Upload your CSV file with water quality parameters
    3. Select a model and get predictions
    
    ### Data Requirements:
    Your CSV should contain water quality measurements such as:
    - pH, Temperature, Dissolved Oxygen
    - BOD, COD, Turbidity
    - Conductivity, Hardness
    - And other chemical parameters
    """)
    
    # show model comparison if available
    ui.show_comparison_chart()

# ============================================
# PREDICTIONS PAGE
# ============================================
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    # file upload
    uploaded_file = ui.file_uploader()
    
    if uploaded_file is not None:
        try:
            # load data
            df = pd.read_csv(uploaded_file)
            ui.show_dataframe(df, "Uploaded Data")
            
            # model selection
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_name = ui.model_selector()
            
            with col2:
                use_tuned = st.checkbox("Use Tuned Model", value=True, 
                                       help="Use model optimized with Grid Search CV")
            
            # predict button
            if st.button("🚀 Predict", type="primary"):
                with st.spinner("Loading model..."):
                    try:
                        # load model and preprocessor
                        model_interface.load_model(model_name, use_tuned=use_tuned)
                        model_interface.load_preprocessor()
                        
                        ui.show_success(f"Loaded {model_name}")
                        
                        # make predictions
                        with st.spinner("Making predictions..."):
                            results = model_interface.predict_from_dataframe(df)
                        
                        # show results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        # create results dataframe
                        results_df = df.copy()
                        results_df['Predicted_Class'] = results['predictions']
                        
                        if results['probabilities']:
                            # add max probability
                            probs = results['probabilities']
                            max_probs = [max(p) for p in probs]
                            results_df['Confidence'] = [f"{p:.2%}" for p in max_probs]
                        
                        # show results
                        st.dataframe(results_df, use_container_width=True)
                        
                        # download button
                        csv = results_df.to_csv(index=False)
                        ui.download_button(
                            csv, 
                            'predictions.csv', 
                            '📥 Download Predictions'
                        )
                        
                        # show summary
                        st.markdown("---")
                        st.subheader("Summary")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Samples", len(results_df))
                        with col2:
                            st.metric("Unique Classes", results_df['Predicted_Class'].nunique())
                        
                        # class distribution
                        st.write("**Predicted Class Distribution:**")
                        class_counts = results_df['Predicted_Class'].value_counts()
                        st.bar_chart(class_counts)
                        
                    except Exception as e:
                        ui.show_error(f"Prediction failed: {str(e)}")
        
        except Exception as e:
            ui.show_error(f"Failed to load data: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # show example format
        with st.expander("📋 See Example Data Format"):
            example_df = pd.DataFrame({
                'pH': [7.2, 6.8, 7.5],
                'Temperature': [25.3, 24.1, 26.2],
                'Dissolved_O2': [6.5, 5.8, 7.2],
                'BOD': [3.2, 4.5, 2.8],
                'COD': [15.3, 18.2, 14.1]
            })
            st.dataframe(example_df)

# ============================================
# MODEL EVALUATION PAGE
# ============================================
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")
    
    # check if models exist
    available_models = model_interface.get_available_models()
    
    if not available_models:
        ui.show_warning("No trained models found. Please train models first.")
    else:
        # model selection
        model_name = st.selectbox("Select Model to Evaluate", available_models)
        
        # load and show metrics
        metrics = model_interface.load_metrics(model_name)
        
        if metrics:
            ui.show_metrics(metrics, model_name)
            
            st.markdown("---")
            
            # show visualizations
            tab1, tab2 = st.tabs(["Confusion Matrix", "Feature Importance"])
            
            with tab1:
                ui.show_confusion_matrix(model_name)
            
            with tab2:
                ui.show_feature_importance(model_name)
        
        else:
            ui.show_warning(f"No evaluation metrics found for {model_name}")
        
        # show comparison
        st.markdown("---")
        ui.show_comparison_chart()

# ============================================
# ABOUT PAGE
# ============================================
elif page == "About":
    st.header("ℹ️ About")
    
    st.markdown("""
    ### Water Quality Classification System
    
    This application uses machine learning to classify water quality based on 
    various physical, chemical, and biological parameters.
    
    ### Project Structure:
    - **Data Processing**: Automated cleaning, validation, and preprocessing
    - **Feature Engineering**: Creation of meaningful features from raw data
    - **Model Training**: Multiple algorithms with hyperparameter tuning
    - **Evaluation**: Comprehensive metrics and visualizations
    - **Deployment**: User-friendly web interface
    
    ### Models Used:
    
    **1. Decision Tree**
    - Simple, interpretable model
    - Good for understanding feature importance
    - Fast training and prediction
    
    **2. Random Forest**
    - Ensemble of multiple decision trees
    - Reduces overfitting
    - High accuracy
    
    **3. XGBoost**
    - Gradient boosting algorithm
    - State-of-the-art performance
    - Handles complex patterns
    
    **4. Logistic Regression**
    - Linear model baseline
    - Fast and efficient
    - Good for linearly separable data
    
    ### Technologies:
    - Python 3.8+
    - scikit-learn
    - XGBoost
    - Streamlit
    - pandas, numpy
    - matplotlib, seaborn
    
    ### Data Privacy:
    - All data processing is done locally
    - No data is stored or transmitted
    - Uploaded files are processed in memory only
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** 2025
    """)

# footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Water Quality Classification System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)