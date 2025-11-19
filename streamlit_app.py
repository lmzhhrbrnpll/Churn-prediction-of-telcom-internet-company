import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Customer Churn Prediction")
st.markdown("""
Fast churn prediction using pre-optimized LightGBM model with important features only.
""")

# Pre-selected important features to reduce dimensionality
IMPORTANT_FEATURES = [
    'tenure', 'monthly_charges', 'total_charges', 'contract', 
    'internet_service_x', 'online_security', 'tech_support',
    'senior_citizen', 'paperless_billing', 'payment_method'
]

@st.cache_data
def load_data():
    """Load and preprocess data quickly"""
    try:
        data = pd.read_csv('Data/train_data.csv')
        
        # Select only important features
        available_features = [f for f in IMPORTANT_FEATURES if f in data.columns]
        available_features.append('churn_value')
        
        data = data[available_features].dropna()
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_fast_model(data):
    """Train model with optimized settings for speed"""
    try:
        # Prepare data
        X = data.drop('churn_value', axis=1)
        y = data['churn_value']
        
        # Quick encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fast SMOTE (reduced sampling)
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Pre-trained model with good default parameters (no grid search)
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # Fast training
        model.fit(X_train_smote, y_train_smote)
        
        # Quick predictions
        y_pred = model.predict(X_test)
        
        return {
            'model': model,
            'label_encoders': label_encoders,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': X.columns.tolist()
        }
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def create_simple_input_form(label_encoders, feature_names):
    """Create simplified input form"""
    st.sidebar.header("🔧 Customer Input")
    
    inputs = {}
    
    # Quick numerical inputs with defaults
    numerical_defaults = {
        'tenure': 12,
        'monthly_charges': 65.0,
        'total_charges': 2000.0
    }
    
    for feature, default in numerical_defaults.items():
        if feature in feature_names:
            if feature == 'tenure':
                inputs[feature] = st.sidebar.slider('Tenure (months)', 0, 72, default)
            else:
                inputs[feature] = st.sidebar.number_input(
                    feature.replace('_', ' ').title(), 
                    min_value=0.0, max_value=10000.0, value=default
                )
    
    # Quick categorical inputs
    categorical_mapping = {
        'contract': ['Month-to-month', 'One year', 'Two year'],
        'internet_service_x': ['DSL', 'Fiber optic', 'No'],
        'online_security': ['No', 'Yes'],
        'tech_support': ['No', 'Yes'],
        'senior_citizen': ['No', 'Yes'],
        'paperless_billing': ['No', 'Yes'],
        'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    }
    
    for feature, options in categorical_mapping.items():
        if feature in feature_names and feature in label_encoders:
            # Use predefined options instead of encoder classes for speed
            inputs[feature] = st.sidebar.selectbox(
                feature.replace('_', ' ').title(), 
                options
            )
    
    return inputs

def main():
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check if 'train_data.csv' exists.")
        return
    
    # Quick stats
    st.subheader("📁 Quick Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Samples", len(data))
    
    with col2:
        churn_rate = data['churn_value'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Train model button
    if st.button("🚀 Train Model (Fast)", type="primary"):
        with st.spinner("Training model quickly..."):
            model_data = train_fast_model(data)
            
        if model_data is not None:
            st.session_state.model_data = model_data
            st.success("Model trained successfully!")
    
    # If model is trained, show quick interface
    if 'model_data' in st.session_state:
        model_data = st.session_state.model_data
        
        # Quick metrics
        st.subheader("📊 Model Performance")
        accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
        f1 = f1_score(model_data['y_test'], model_data['y_pred'])
        
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("F1-Score", f"{f1:.3f}")
        
        # Quick prediction interface
        st.subheader("🔮 Quick Prediction")
        inputs = create_simple_input_form(model_data['label_encoders'], model_data['feature_names'])
        
        if st.button("Predict", type="primary"):
            try:
                # Fast prediction preparation
                input_df = pd.DataFrame([inputs])
                
                # Quick encoding
                for col, encoder in model_data['label_encoders'].items():
                    if col in input_df.columns:
                        input_df[col] = encoder.transform([input_df[col].iloc[0]])[0]
                
                # Ensure feature order
                input_df = input_df.reindex(columns=model_data['feature_names'], fill_value=0)
                
                # Fast prediction
                prediction = model_data['model'].predict(input_df)[0]
                probability = model_data['model'].predict_proba(input_df)[0]
                
                # Quick results
                st.subheader("🎯 Results")
                if prediction == 1:
                    st.error(f"**CHURN RISK: HIGH** ({probability[1]:.1%})")
                    st.progress(probability[1])
                else:
                    st.success(f"**CHURN RISK: LOW** ({probability[0]:.1%})")
                    st.progress(probability[0])
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    else:
        st.info("👆 Click 'Train Model' to start (this will be fast!)")

if __name__ == "__main__":
    main()
