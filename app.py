#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1. Imports
# ==============================================================================
import streamlit as st
import pandas as pd
import joblib

# 2. Functions
# ==============================================================================

@st.cache_resource
def load_artifacts():
    """
    Load the required objects for prediction.
    Uses Streamlit cache so artifacts are not reloaded on every interaction.
    """
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        model = joblib.load('attrition_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        print("✅ Artifacts (preprocessor, model, label_encoder) loaded successfully.")
        return preprocessor, model, label_encoder
    except FileNotFoundError:
        st.error("Error: Please ensure 'preprocessor.joblib', 'attrition_model.joblib', and 'label_encoder.joblib' are in the same directory.")
        return None, None, None

def create_input_form(X_columns):
    """
    Create an input form in the Streamlit sidebar.

    Args:
        X_columns (list): List of column names from training data to ensure consistency.

    Returns:
        pandas.DataFrame: DataFrame containing a single row of user input.
    """
    st.sidebar.header("Enter Employee Data:")
    
    # User input
    overtime = st.sidebar.selectbox('OverTime', ['Yes', 'No'])
    monthly_income = st.sidebar.number_input('Monthly Income', min_value=1000, max_value=20000, value=6500, step=100)
    age = st.sidebar.number_input('Age', min_value=18, max_value=60, value=37, step=1)
    total_working_years = st.sidebar.number_input('Total Working Years', min_value=0, max_value=40, value=11, step=1)
    job_level = st.sidebar.number_input('Job Level', min_value=1, max_value=5, value=2, step=1)
    
    # Ensure correct data type for JobInvolvement
    job_involvement = int(st.sidebar.selectbox('Job Involvement', [1, 2, 3, 4], index=2))
    
    # Create dictionary from input
    input_data = {
        'Age': age,
        'BusinessTravel': 'Travel_Rarely',
        'DailyRate': 802,
        'Department': 'Research & Development',
        'DistanceFromHome': 9,
        'Education': 3,
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 3,
        'Gender': 'Male',
        'HourlyRate': 66,
        'JobInvolvement': job_involvement, # Ensure this is integer
        'JobLevel': job_level,
        'JobRole': 'Research Scientist',
        'JobSatisfaction': 3,
        'MaritalStatus': 'Married',
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14235,
        'NumCompaniesWorked': 2,
        'OverTime': overtime,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 3,
        'YearsAtCompany': 7,
        'YearsInCurrentRole': 4,
        'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 4,
    }

    # Create DataFrame from dictionary
    input_df = pd.DataFrame([input_data])
    
    # Return DataFrame with columns in the same order as training
    return input_df[X_columns]


# 3. Main Application
# ==============================================================================

# Page configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Load artifacts
preprocessor, model, label_encoder = load_artifacts()

# App title
st.title("👨‍💼 Employee Attrition Prediction App")
st.write("This app uses a machine learning model to predict the likelihood of an employee leaving (attrition).")

if preprocessor and model and label_encoder:
    # Get column names required by the preprocessor
    try:
        # For scikit-learn >= 1.2
        X_cols = preprocessor.get_feature_names_out()
    except AttributeError:
        # For scikit-learn < 1.2 (fallback)
        cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(preprocessor.feature_names_in_[preprocessor.transformers_[1][2]])
        num_cols = preprocessor.feature_names_in_[preprocessor.transformers_[0][2]]
        X_cols = list(num_cols) + list(cat_cols)
    
    # Raw columns before processing (needed for input form)
    raw_cols = preprocessor.feature_names_in_

    # Show input form in sidebar
    input_df = create_input_form(raw_cols)

    st.subheader("Entered Employee Data:")
    st.dataframe(input_df) 

    # Prediction button
    if st.button("🔮 Predict Attrition Risk", type="primary"):
        # Make prediction
        input_processed = preprocessor.transform(input_df)
        prediction_proba = model.predict_proba(input_processed)
        
        # Probability for 'Yes' (Attrition)
        prob_attrition = prediction_proba[0, 1]
        
        st.subheader("Prediction Result:")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Attrition Risk Score", value=f"{prob_attrition:.2%}")
            
            if prob_attrition > 0.5:
                st.error("Status: **High Risk of Attrition**")
            elif prob_attrition > 0.3:
                st.warning("Status: **Medium Risk (Needs Attention)**")
            else:
                st.success("Status: **Likely to Stay**")

        with col2:
            st.write("**Score Interpretation:**")
            st.info(f"""
            This score indicates the probability that the employee will leave (attrition).
            - **< 30%**: Low risk.
            - **30% - 50%**: Medium risk. Retention actions are recommended.
            - **> 50%**: High risk. Immediate attention required.

            *Based on evaluation, this model may not capture all attrition cases (low recall), so scores above 30% should be considered with caution.*
            """)
else:
    st.stop()