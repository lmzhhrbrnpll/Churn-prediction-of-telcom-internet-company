import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
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
This app predicts customer churn using a LightGBM model with SMOTE and hyperparameter tuning.
Only the most important features are shown for input.
""")

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        data = pd.read_csv('train_data.csv')
        
        # Select important features based on domain knowledge and correlation analysis
        important_features = [
            'tenure', 'monthly_charges', 'total_charges', 'contract', 'internet_service_x',
            'online_security', 'tech_support', 'senior_citizen', 'partner', 'dependents',
            'paperless_billing', 'payment_method', 'total_revenue', 'cltv', 'satisfaction_score'
        ]
        
        important_features = [f for f in important_features if f in data.columns]
        important_features.append('churn_value')
        
        data = data[important_features]
        
        # Handle missing values
        data = data.dropna()
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(data):
    """Train the LightGBM model with SMOTE and GridSearchCV"""
    try:
        # Prepare features and target
        X = data.drop('churn_value', axis=1)
        y = data['churn_value']
        
        # Encode categorical variables
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
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Define parameter grid for LightGBM
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        # Train with GridSearchCV
        lgbm = LGBMClassifier(random_state=42, verbose=-1)
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_smote, y_train_smote)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        return {
            'model': best_model,
            'label_encoders': label_encoders,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_names': X.columns.tolist(),
            'best_params': grid_search.best_params_
        }
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def create_input_form(label_encoders, feature_names):
    """Create input form for user predictions"""
    st.sidebar.header("🔧 Customer Input Parameters")
    
    inputs = {}
    
    # Numerical features
    if 'tenure' in feature_names:
        inputs['tenure'] = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    
    if 'monthly_charges' in feature_names:
        inputs['monthly_charges'] = st.sidebar.number_input('Monthly Charges ($)', 
                                                         min_value=0.0, max_value=200.0, value=50.0)
    
    if 'total_charges' in feature_names:
        inputs['total_charges'] = st.sidebar.number_input('Total Charges ($)', 
                                                        min_value=0.0, max_value=10000.0, value=1000.0)
    
    if 'total_revenue' in feature_names:
        inputs['total_revenue'] = st.sidebar.number_input('Total Revenue ($)', 
                                                       min_value=0.0, max_value=20000.0, value=2000.0)
    
    if 'cltv' in feature_names:
        inputs['cltv'] = st.sidebar.number_input('Customer Lifetime Value', 
                                              min_value=0, max_value=10000, value=4000)
    
    if 'satisfaction_score' in feature_names:
        inputs['satisfaction_score'] = st.sidebar.slider('Satisfaction Score (1-5)', 1, 5, 3)
    
    # Categorical features
    if 'contract' in feature_names and 'contract' in label_encoders:
        contract_options = label_encoders['contract'].classes_
        inputs['contract'] = st.sidebar.selectbox('Contract Type', contract_options)
    
    if 'internet_service_x' in feature_names and 'internet_service_x' in label_encoders:
        internet_options = label_encoders['internet_service_x'].classes_
        inputs['internet_service_x'] = st.sidebar.selectbox('Internet Service', internet_options)
    
    if 'online_security' in feature_names and 'online_security' in label_encoders:
        security_options = label_encoders['online_security'].classes_
        inputs['online_security'] = st.sidebar.selectbox('Online Security', security_options)
    
    if 'tech_support' in feature_names and 'tech_support' in label_encoders:
        tech_options = label_encoders['tech_support'].classes_
        inputs['tech_support'] = st.sidebar.selectbox('Tech Support', tech_options)
    
    if 'senior_citizen' in feature_names and 'senior_citizen' in label_encoders:
        senior_options = label_encoders['senior_citizen'].classes_
        inputs['senior_citizen'] = st.sidebar.selectbox('Senior Citizen', senior_options)
    
    if 'partner' in feature_names and 'partner' in label_encoders:
        partner_options = label_encoders['partner'].classes_
        inputs['partner'] = st.sidebar.selectbox('Partner', partner_options)
    
    if 'dependents' in feature_names and 'dependents' in label_encoders:
        dependents_options = label_encoders['dependents'].classes_
        inputs['dependents'] = st.sidebar.selectbox('Dependents', dependents_options)
    
    if 'paperless_billing' in feature_names and 'paperless_billing' in label_encoders:
        paperless_options = label_encoders['paperless_billing'].classes_
        inputs['paperless_billing'] = st.sidebar.selectbox('Paperless Billing', paperless_options)
    
    if 'payment_method' in feature_names and 'payment_method' in label_encoders:
        payment_options = label_encoders['payment_method'].classes_
        inputs['payment_method'] = st.sidebar.selectbox('Payment Method', payment_options)
    
    return inputs

def main():
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check if 'train_data.csv' exists.")
        return
    
    # Display dataset info
    st.subheader("📁 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(data))
    
    with col2:
        churn_rate = data['churn_value'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        st.metric("Features Used", len(data.columns) - 1)
    
    # Train or load model
    if st.button("🔄 Train Model", type="primary"):
        with st.spinner("Training model with SMOTE and GridSearchCV..."):
            model_data = train_model(data)
            
        if model_data is not None:
            st.session_state.model_data = model_data
            st.success("Model trained successfully!")
    
    # If model is trained, show results and prediction interface
    if 'model_data' in st.session_state:
        model_data = st.session_state.model_data
        
        # Display model information
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best Parameters:**")
            st.json(model_data['best_params'])
        
        with col2:
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
            f1 = f1_score(model_data['y_test'], model_data['y_pred'])
            precision = precision_score(model_data['y_test'], model_data['y_pred'])
            recall = recall_score(model_data['y_test'], model_data['y_pred'])
            
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
        
        # Create input form for predictions
        st.subheader("🔮 Make Predictions")
        inputs = create_input_form(model_data['label_encoders'], model_data['feature_names'])
        
        if st.button("Predict Churn", type="primary"):
            try:
                # Prepare input data
                input_df = pd.DataFrame([inputs])
                
                # Encode categorical variables
                for col, encoder in model_data['label_encoders'].items():
                    if col in input_df.columns:
                        # Handle unseen labels
                        input_df[col] = input_df[col].apply(
                            lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                        )
                        input_df[col] = encoder.transform(input_df[col])
                
                # Ensure all features are present and in correct order
                for feature in model_data['feature_names']:
                    if feature not in input_df.columns:
                        input_df[feature] = 0  # Default value for missing features
                
                input_df = input_df[model_data['feature_names']]
                
                # Make prediction
                prediction = model_data['model'].predict(input_df)[0]
                probability = model_data['model'].predict_proba(input_df)[0]
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error(f"**Prediction: CHURN** ⚠️")
                    else:
                        st.success(f"**Prediction: NO CHURN** ✅")
                
                with col2:
                    st.metric("Probability of Churn", f"{probability[1]:.3f}")
                    st.metric("Probability of No Churn", f"{probability[0]:.3f}")
                
                # Feature importance (if available)
                if hasattr(model_data['model'], 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    
                    importance_df = pd.DataFrame({
                        'feature': model_data['feature_names'],
                        'importance': model_data['model'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('feature')['importance'].head(10))
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    else:
        st.info("👆 Click the 'Train Model' button to start training the prediction model.")

if __name__ == "__main__":
    main()
