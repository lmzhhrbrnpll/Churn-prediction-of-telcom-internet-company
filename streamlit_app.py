import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
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

# --- MODEL TRAINING WITH LIGHTGBM AND SMOTE ---
@st.cache_resource
def train_lightgbm_model(df_ml):
    """Train LightGBM model with hyperparameter tuning and SMOTE."""

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

    # LightGBM hyperparameter tuning
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        lgb_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
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
            y_pred, y_pred_proba, grid_search.best_params_, metrics)

# --- MAIN APP ---
st.title("📊 Customer Churn Analysis & Prediction Dashboard")
st.markdown("**Model: LightGBM with Hyperparameter Tuning + SMOTE**")

# Create tabs for Data Overview and Prediction
tab1, tab2 = st.tabs(["📈 🔍 Data Overview", "🤖 Churn Prediction Model")

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

    # Filter for other numerical columns (select top 5 for simplicity)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'churn_value'][:5]
    
    for col in numerical_cols:
        if col in df.columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = (min_val, max_val)
            selected_range = st.sidebar.slider(
                f"Select {col} Range",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    # --- FILTERING THE DATAFRAME ---
    df_selection = df.copy()

    # Apply churn filter
    df_selection = df_selection[df_selection["churn_value"].isin(churn_status)]

    # Apply numerical filters
    for col in numerical_cols:
        if col in df_selection.columns:
            min_val, max_val = selected_range
            df_selection = df_selection[(df_selection[col] >= min_val) & (df_selection[col] <= max_val)]

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

        # Correlation matrix
        st.subheader("📊 Correlation Matrix")
        corr_matrix = df.corr()

        # --- DISPLAY RAW DATA ---
        with st.expander("View Filtered Data"):
            st.dataframe(df_selection)
            st.markdown(f"**Data Dimensions:** {df_selection.shape[0]} rows, {df_selection.shape[1]} columns")

with tab2:
    # --- CHURN PREDICTION MODEL ---
    st.header("🤖 Customer Churn Prediction Model")
    st.markdown("""
    This section uses a tuned LightGBM classifier with SMOTE to handle class imbalance for customer churn prediction.
    Target variable: **churn_value**
    """)

    # Preprocess data for ML
    df_ml = preprocess_data(df)

    # Display preprocessing info
    with st.expander("Data Preprocessing Details"):
        st.write("**Data Shape:**", df_ml.shape)
        st.write("**Class Distribution:**")
        st.write(df_ml['churn_value'].value_counts())

    # Train model
    with st.spinner("Training LightGBM model with SMOTE and hyperparameter tuning..."):
        (best_model, scaler, smote, X_train, X_test, y_train, y_test,
         y_pred, y_pred_proba, best_params, metrics) = train_lightgbm_model(df_ml)

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
    st.subheader("⚙️ Best Hyperparameters")
    st.json(best_params)

    # --- PREDICTION INTERFACE ---
    st.subheader("🎯 Make Predictions")

    st.markdown("""
    Enter customer details below to predict churn probability.
    """)

    # Get feature names for input
    feature_columns = X_train.columns.tolist()

    # Create input form
    with st.form("prediction_form"):
        st.write("### Customer Details")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        input_data = {}

        with col1:
            for feature in feature_columns[:len(feature_columns)//2]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )

        with col2:
            for feature in feature_columns[len(feature_columns)//2:]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure correct column order
            input_df = input_df[feature_columns]

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
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
        ax.set_title('Top 10 Feature Importance')
        ax.set_xlabel('Importance Score')
        st.pyplot(fig)
    else:
        # Fallback: display as bar chart using streamlit
        st.bar_chart(feature_importance.head(10).set_index('feature'))

    # Display feature importance table
    with st.expander("View All Feature Importance"):
        st.dataframe(feature_importance)

    # Classification Report
    st.subheader("📋 Classification Report")
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    st.dataframe(clf_report_df)


st.markdown("---")
st.write("Customer Churn Analysis & Prediction Dashboard | Model: LightGBM with SMOTE")
