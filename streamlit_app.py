import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
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
        df = pd.read_csv("train_data.csv")
        st.success(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# --- DATA PREPROCESSING FOR MODEL ---
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning model - handle categorical variables."""
    df_ml = df.copy()

    # Convert categorical variables to numeric
    categorical_columns = ['premium_tech_support', 'dependents']
    
    for col in categorical_columns:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map({'Yes': 1, 'No': 0})
    
    # Ensure all columns are numeric
    non_numeric_cols = df_ml.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        st.warning(f"Kolom non-numerik ditemukan dan akan dihapus: {list(non_numeric_cols)}")
        df_ml = df_ml.drop(columns=non_numeric_cols)

    # Handle missing values if any
    if df_ml.isnull().sum().sum() > 0:
        numeric_columns = df_ml.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_ml[col].isnull().sum() > 0:
                df_ml[col].fillna(df_ml[col].median(), inplace=True)

    return df_ml

# --- OPTIMIZED MODEL TRAINING ---
@st.cache_resource
def train_lightgbm_model_optimized(df_ml):
    """Train LightGBM model with optimized hyperparameters and SMOTE."""
    
    # Prepare features and target
    X = df_ml.drop('churn_value', axis=1)
    y = df_ml['churn_value']

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

    # OPTIMIZED: Use fixed hyperparameters instead of GridSearchCV
    # These are commonly good parameters for LightGBM with SMOTE
    best_params = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }

    # Train model with optimized parameters
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(X_train_scaled, y_train_resampled)

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
            y_pred, y_pred_proba, best_params, metrics)

# --- MAIN APP ---
st.title("📊 Customer Churn Analysis & Prediction Dashboard")
st.markdown("**Model: LightGBM with Optimized Hyperparameters + SMOTE**")

# Create tabs for Data Overview and Prediction
tab1, tab2 = st.tabs(["📈 🔍 Data Overview", "🤖 Churn Prediction Model"])

with tab1:
    # --- Data Overview CONTENT ---
    st.markdown("""
    This application performs data overview on the preprocessed Customer Churn dataset.
    Use the filters in the sidebar to explore customer behavior and churn patterns.
    """)

    # --- SIDEBAR FOR FILTERS ---
    st.sidebar.header("Filter Customers")

    # Filter for Churn status
    churn_status = st.sidebar.multiselect(
        "Select Churn Status",
        options=df["churn_value"].unique(),
        default=df["churn_value"].unique()
    )

    # Filter for other numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'churn_value'][:5]
    
    # Apply filters to dataframe
    df_selection = df.copy()
    df_selection = df_selection[df_selection["churn_value"].isin(churn_status)]

    # Display error message if no data is selected
    if df_selection.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
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
        for i, col in enumerate(numerical_cols[:3]):
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

        # --- DISPLAY RAW DATA ---
        with st.expander("View Filtered Data"):
            st.dataframe(df_selection)
            st.markdown(f"**Data Dimensions:** {df_selection.shape[0]} rows, {df_selection.shape[1]} columns")

with tab2:
    # --- CHURN PREDICTION MODEL ---
    st.header("🤖 Customer Churn Prediction Model")
    st.markdown("""
    This section uses an optimized LightGBM classifier with SMOTE to handle class imbalance for customer churn prediction.
    Target variable: **churn_value**
    """)

    # Preprocess data for ML
    df_ml = preprocess_data(df)

    # Display preprocessing info
    with st.expander("Data Preprocessing Details"):
        st.write("**Data Shape:**", df_ml.shape)
        st.write("**Class Distribution:**")
        st.write(df_ml['churn_value'].value_counts())
        st.write("**Features Used:**", list(df_ml.columns))

    # Train model with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Preparing data...")
    progress_bar.progress(20)
    
    status_text.text("Applying SMOTE...")
    progress_bar.progress(40)
    
    status_text.text("Training LightGBM model with optimized parameters...")
    
    # Train model
    try:
        (best_model, scaler, smote, X_train, X_test, y_train, y_test,
         y_pred, y_pred_proba, best_params, metrics) = train_lightgbm_model_optimized(df_ml)
        
        progress_bar.progress(80)
        status_text.text("Making predictions and calculating metrics...")
        progress_bar.progress(100)
        status_text.text("")

        st.success("Model training completed!")

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

        # Display best parameters
        st.subheader("⚙️ Optimized Hyperparameters")
        st.json(best_params)

        # Get all features for prediction form
        all_features = X_train.columns.tolist()
        
        # Display features info
        with st.expander("📋 Features Used for Prediction"):
            st.write("The following features are used for churn prediction:")
            for i, feature in enumerate(all_features, 1):
                st.write(f"{i}. {feature}")

        # --- PREDICTION INTERFACE ---
        st.subheader("🎯 Make Predictions")

        st.markdown("""
        Enter customer details below to predict churn probability. All available features are shown.
        """)

        # Create input form with all available features
        with st.form("prediction_form"):
            st.write("### Customer Details")
            
            # Group features into categories for better organization
            input_data = {}
            
            # Service Usage & Demographics
            st.write("#### 📊 Service Usage & Demographics")
            demo_col1, demo_col2, demo_col3 = st.columns(3)
            
            with demo_col1:
                if 'monthly_charges' in all_features:
                    min_val = float(df['monthly_charges'].min())
                    max_val = float(df['monthly_charges'].max())
                    default_val = float(df['monthly_charges'].median())
                    input_data['monthly_charges'] = st.slider(
                        "Monthly Charges",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                
                if 'tenure' in all_features:
                    min_val = float(df['tenure'].min())
                    max_val = float(df['tenure'].max())
                    default_val = float(df['tenure'].median())
                    input_data['tenure'] = st.slider(
                        "Tenure",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                    
            with demo_col2:
                if 'age' in all_features:
                    min_val = float(df['age'].min())
                    max_val = float(df['age'].max())
                    default_val = float(df['age'].median())
                    input_data['age'] = st.slider(
                        "Age",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                
                if 'total_population' in all_features:
                    min_val = float(df['total_population'].min())
                    max_val = float(df['total_population'].max())
                    default_val = float(df['total_population'].median())
                    input_data['total_population'] = st.slider(
                        "Total Population",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                    
            with demo_col3:
                if 'number_of_referrals' in all_features:
                    min_val = float(df['number_of_referrals'].min())
                    max_val = float(df['number_of_referrals'].max())
                    default_val = float(df['number_of_referrals'].median())
                    input_data['number_of_referrals'] = st.slider(
                        "Number of Referrals",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                
                if 'cltv' in all_features:
                    min_val = float(df['cltv'].min())
                    max_val = float(df['cltv'].max())
                    default_val = float(df['cltv'].median())
                    input_data['cltv'] = st.slider(
                        "CLTV",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )

            # Service Features
            st.write("#### 🔧 Service Features")
            service_col1, service_col2, service_col3 = st.columns(3)
            
            with service_col1:
                if 'avg_monthly_gb_download' in all_features:
                    min_val = float(df['avg_monthly_gb_download'].min())
                    max_val = float(df['avg_monthly_gb_download'].max())
                    default_val = float(df['avg_monthly_gb_download'].median())
                    input_data['avg_monthly_gb_download'] = st.slider(
                        "Avg Monthly GB Download",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                
                if 'satisfaction_score' in all_features:
                    min_val = float(df['satisfaction_score'].min())
                    max_val = float(df['satisfaction_score'].max())
                    default_val = float(df['satisfaction_score'].median())
                    input_data['satisfaction_score'] = st.slider(
                        "Satisfaction Score",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                    
            with service_col2:
                if 'premium_tech_support' in all_features:
                    input_data['premium_tech_support'] = st.selectbox(
                        "Premium Tech Support",
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes"
                    )
                
                if 'dependents' in all_features:
                    input_data['dependents'] = st.selectbox(
                        "Has Dependents",
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
                
                # Update with user input for available features
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

        # Classification Report
        st.subheader("📋 Classification Report")
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        st.dataframe(clf_report_df)

    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.info("Pastikan semua kolom dalam dataset adalah numerik atau sudah di-encode dengan benar.")

st.markdown("---")
st.write("Customer Churn Analysis & Prediction Dashboard | Model: LightGBM with SMOTE")
