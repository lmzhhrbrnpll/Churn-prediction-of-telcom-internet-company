import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try importing matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    st.warning("Matplotlib tidak tersedia. Beberapa visualisasi mungkin tidak berfungsi.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads the preprocessed customer churn dataset."""
    try:
        df = pd.read_csv("Data/train_data.csv")
        st.success(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# --- DATA PREPROCESSING FOR MODEL ---
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning model - simplified for clean data."""
    df_ml = df.copy()

    # Handle missing values if any
    if df_ml.isnull().sum().sum() > 0:
        numeric_columns = df_ml.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_ml[col].isnull().sum() > 0:
                df_ml[col].fillna(df_ml[col].median(), inplace=True)

    return df_ml

# --- MODEL TRAINING WITH LIGHTGBM AND SMOTE WITH HYPERPARAMETER TUNING ---
@st.cache_resource
def train_lightgbm_with_tuning(_df_ml):
    """Train LightGBM model with hyperparameter tuning and SMOTE."""
    
    # Prepare features and target
    X = _df_ml.drop('churn_value', axis=1)
    y = _df_ml['churn_value']

    # Split the data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # LightGBM hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)

    # Use Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 for speed

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        lgb_model, param_grid, cv=cv, scoring='roc_auc', 
        n_jobs=-1, verbose=0, return_train_score=True
    )

    grid_search.fit(X_train_scaled, y_train_resampled)

    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score
    }

    return (best_model, scaler, smote, X_train, X_test, y_train, y_test,
            y_pred, y_pred_proba, grid_search.best_params_, metrics, grid_search)

# --- GET IMPORTANT FEATURES ---
def get_important_features(model, feature_names, top_n=15):
    """Get top N most important features from the trained model."""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n)['feature'].tolist()

# --- MAIN APP ---
st.title("📊 Customer Churn Analysis & Prediction Dashboard")
st.markdown("**Model: LightGBM with Hyperparameter Tuning + SMOTE**")

# Create tabs for Data Overview and Prediction
tab1, tab2 = st.tabs(["📈 🔍 Data Overview", "🤖 Churn Prediction Model"])

with tab1:
    # --- Data Overview CONTENT ---
    st.markdown("""
    This application performs data overview on the preprocessed Customer Churn dataset.
    """)

    # Use the full dataset without filtering
    df_selection = df.copy()

    # --- MAIN PAGE CONTENT ---
    st.subheader("📈 Key Metrics")

    # --- DISPLAY KEY METRICS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        total_customers = df_selection.shape[0]
        st.metric(label="Total Customers", value=total_customers)

    with col2:
        churn_rate = (df_selection["churn_value"].sum() / total_customers) * 100
        st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")

    with col3:
        st.metric(label="Churned Customers", value=df_selection["churn_value"].sum())

    # Additional metrics for numerical columns
    col4, col5, col6 = st.columns(3)

    # Display average values for top 3 numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'churn_value'][:3]
    
    for i, col in enumerate(numerical_cols):
        with [col4, col5, col6][i]:
            avg_value = round(df_selection[col].mean(), 3)
            st.metric(label=f"Avg. {col}", value=avg_value)

    st.markdown("---")
    
    # --- DATA SUMMARY ---
    st.subheader("📋 Data Summary")

    # Display some statistics
    col9, col10 = st.columns(2)

    with col9:
        st.write("**Churn Statistics:**")
        st.write(f"- Churned Customers: {df_selection['churn_value'].sum()}")
        st.write(f"- Non-Churned Customers: {len(df_selection) - df_selection['churn_value'].sum()}")
        st.write(f"- Churn Rate: {churn_rate:.2f}%")

    with col10:
        st.write("**Dataset Information:**")
        st.write(f"- Total Features: {df_selection.shape[1]}")
        st.write(f"- Total Samples: {df_selection.shape[0]}")
        st.write(f"- Numerical Features: {len(df_selection.select_dtypes(include=[np.number]).columns)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    
        # Data types
        st.write("**Data Types:**")
        dtype_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_info)

    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

    # --- VISUALIZATIONS ---
    st.subheader("📊 Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Churn distribution
        st.subheader("Churn Distribution")
        churn_counts = df_selection["churn_value"].value_counts()
        churn_df = pd.DataFrame({
            'Churn Status': ['Not Churned', 'Churned'],
            'Count': churn_counts.values
        })
        st.bar_chart(churn_df.set_index('Churn Status'))

    with viz_col2:
        # Show relationship between a numerical feature and churn
        if len(numerical_cols) > 0:
            numerical_feature = numerical_cols[0]
            st.subheader(f"Avg. {numerical_feature} by Churn Status")
            avg_by_churn = df_selection.groupby('churn_value')[numerical_feature].mean()
            st.bar_chart(avg_by_churn)

    # --- DISPLAY RAW DATA ---
    with st.expander("View Raw Data"):
        st.dataframe(df_selection)
        st.markdown(f"**Data Dimensions:** {df_selection.shape[0]} rows, {df_selection.shape[1]} columns")

with tab2:
    # --- CHURN PREDICTION MODEL ---
    st.header("🤖 Customer Churn Prediction Model")
    st.markdown("""
    This section uses a **tuned LightGBM classifier with SMOTE and GridSearchCV** to handle class imbalance for customer churn prediction.
    Target variable: **churn_value**
    """)

    # Preprocess data for ML
    df_ml = preprocess_data(df)

    # Display preprocessing info
    with st.expander("Data Preprocessing Details"):
        st.write("**Data Shape:**", df_ml.shape)
        st.write("**Class Distribution:**")
        st.write(df_ml['churn_value'].value_counts())
        st.write("**Preprocessing Steps:**")
        st.write("- SMOTE for class imbalance handling")
        st.write("- StandardScaler for feature scaling")
        st.write("- GridSearchCV for hyperparameter tuning")

    # Add a button to train the model
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    if not st.session_state.model_trained:
        if st.button("🚀 Train Model with Hyperparameter Tuning", type="primary"):
            with st.spinner("Training LightGBM model with SMOTE and GridSearchCV hyperparameter tuning... This may take a few minutes."):
                try:
                    (best_model, scaler, smote, X_train, X_test, y_train, y_test,
                     y_pred, y_pred_proba, best_params, metrics, grid_search) = train_lightgbm_with_tuning(df_ml)
                    
                    # Store in session state
                    st.session_state.update({
                        'best_model': best_model,
                        'scaler': scaler,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'best_params': best_params,
                        'metrics': metrics,
                        'grid_search': grid_search,
                        'model_trained': True
                    })
                    st.success("Model training with hyperparameter tuning completed!")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        else:
            st.info("Click the button above to train the model with hyperparameter tuning.")
            st.stop()
    else:
        # Load from session state
        best_model = st.session_state.best_model
        scaler = st.session_state.scaler
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_pred_proba = st.session_state.y_pred_proba
        best_params = st.session_state.best_params
        metrics = st.session_state.metrics
        grid_search = st.session_state.grid_search

    # Display model performance
    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    with col2:
        st.metric("AUC Score", f"{metrics['auc_score']:.3f}")

    with col3:
        st.metric("Precision", f"{metrics['precision']:.3f}")

    with col4:
        st.metric("Recall", f"{metrics['recall']:.3f}")

    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

    with col6:
        st.metric("Training Samples", X_train.shape[0])

    with col7:
        st.metric("Test Samples", X_test.shape[0])

    with col8:
        st.metric("Features", X_train.shape[1])

    # Display best parameters from hyperparameter tuning
    st.subheader("⚙️ Best Hyperparameters from GridSearchCV")
    st.json(best_params)

    # Display information about the tuning process
    with st.expander("🔧 Hyperparameter Tuning Details"):
        st.write("**GridSearchCV Configuration:**")
        st.write(f"- Number of parameter combinations tested: {len(grid_search.cv_results_['params'])}")
        st.write(f"- Cross-validation folds: 3")
        st.write(f"- Scoring metric: ROC AUC")
        st.write(f"- Best cross-validation score: {grid_search.best_score_:.4f}")
        
        st.write("**Parameter Grid:**")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        st.json(param_grid)

    # Get important features for prediction form
    important_features = get_important_features(best_model, X_train.columns.tolist(), top_n=12)
    
    # Display important features info
    with st.expander("📋 Important Features for Prediction"):
        st.write("The following features are the most important for churn prediction:")
        for i, feature in enumerate(important_features, 1):
            st.write(f"{i}. {feature}")

    # --- PREDICTION INTERFACE ---
    st.subheader("🎯 Make Predictions")

    st.markdown("""
    Enter customer details below to predict churn probability. Only the most important features are shown.
    """)

    # Create input form with only important features
    with st.form("prediction_form"):
        st.write("### Customer Details (Important Features Only)")
        
        input_data = {}
        
        # Organize features into categories
        st.write("#### 📊 Customer Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            if 'age' in important_features:
                min_val = float(df['age'].min())
                max_val = float(df['age'].max())
                default_val = float(df['age'].median())
                input_data['age'] = st.slider(
                    "Age",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            
            if 'tenure' in important_features:
                min_val = float(df['tenure'].min())
                max_val = float(df['tenure'].max())
                default_val = float(df['tenure'].median())
                input_data['tenure'] = st.slider(
                    "Tenure (months)",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
                
        with info_col2:
            if 'monthly_charges' in important_features:
                min_val = float(df['monthly_charges'].min())
                max_val = float(df['monthly_charges'].max())
                default_val = float(df['monthly_charges'].median())
                input_data['monthly_charges'] = st.slider(
                    "Monthly Charges",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1.0
                )

        st.write("#### 🔧 Service Features")
        service_col1, service_col2 = st.columns(2)
        
        with service_col1:
            # Binary features
            binary_features = [f for f in important_features if f in ['online_security', 'online_backup', 'device_protection', 
                                                                     'premium_tech_support', 'streaming_tv', 'streaming_movies',
                                                                     'paperless_billing', 'multiple_lines', 'unlimited_data']]
            
            for feature in binary_features[:4]:  # Show first 4
                input_data[feature] = st.selectbox(
                    feature.replace('_', ' ').title(),
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                )
                
        with service_col2:
            for feature in binary_features[4:8]:  # Show next 4
                if feature in important_features:
                    input_data[feature] = st.selectbox(
                        feature.replace('_', ' ').title(),
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                )

        st.write("#### 📞 Contract & Payment")
        contract_col1, contract_col2 = st.columns(2)
        
        with contract_col1:
            contract_features = [f for f in important_features if 'contract' in f or 'payment' in f]
            for feature in contract_features[:2]:
                input_data[feature] = st.selectbox(
                    feature.replace('_', ' ').title(),
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                )
                
        with contract_col2:
            for feature in contract_features[2:4]:
                if feature in important_features:
                    input_data[feature] = st.selectbox(
                        feature.replace('_', ' ').title(),
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            # Create complete input DataFrame with all features
            complete_input = {}
            
            # Fill all features with median values first
            for feature in X_train.columns:
                complete_input[feature] = float(X_train[feature].median())
            
            # Update with user input for important features
            for feature in input_data:
                if feature in complete_input:
                    complete_input[feature] = input_data[feature]
            
            # Create DataFrame
            input_df = pd.DataFrame([complete_input])
            
            # Ensure correct column order
            input_df = input_df[X_train.columns]

            # Scale the input
            input_scaled = scaler.transform(input_df)

            # Make prediction
            churn_probability = best_model.predict_proba(input_scaled)[0, 1]
            churn_prediction = best_model.predict(input_scaled)[0]

            # Display results
            st.subheader("🎯 Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Churn Probability", f"{churn_probability:.3f}")
                st.metric("Confidence", f"{max(churn_probability, 1-churn_probability):.3f}")

            with result_col2:
                if churn_prediction == 1:
                    st.error("Prediction: **HIGH RISK OF CHURN** ⚠️")
                    st.write("This customer is likely to churn. Consider retention strategies.")
                else:
                    st.success("Prediction: **LOW RISK OF CHURN** ✅")
                    st.write("This customer is likely to stay.")

            # Probability gauge
            st.write("**Churn Probability Gauge:**")
            st.progress(float(churn_probability))
            st.caption(f"Churn likelihood: {churn_probability*100:.1f}%")
            
            # Show interpretation
            with st.expander("💡 Interpretation Guide"):
                st.write("""
                **Churn Probability Interpretation:**
                - **0-30%**: Low risk - Standard customer engagement
                - **30-60%**: Medium risk - Monitor and proactive engagement
                - **60-80%**: High risk - Implement retention strategies
                - **80-100%**: Very high risk - Immediate intervention needed
                """)

    # Confusion Matrix
    st.subheader("📈 Confusion Matrix")
    if matplotlib_available:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    else:
        # Fallback: display confusion matrix as table
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm,
                           index=['Actual Not Churn', 'Actual Churn'],
                           columns=['Predicted Not Churn', 'Predicted Churn'])
        st.dataframe(cm_df)

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    if matplotlib_available:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(12), x='importance', y='feature', ax=ax)
        ax.set_title('Top 12 Feature Importance')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
    else:
        # Fallback: display as bar chart using streamlit
        st.bar_chart(feature_importance.head(12).set_index('feature'))

    # Classification Report
    st.subheader("📋 Classification Report")
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    st.dataframe(clf_report_df)

    # Add retrain button
    if st.button("🔄 Retrain Model with Hyperparameter Tuning"):
        st.session_state.model_trained = False
        st.rerun()

st.markdown("---")
st.write("Customer Churn Analysis & Prediction Dashboard | Model: LightGBM with Hyperparameter Tuning + SMOTE")
