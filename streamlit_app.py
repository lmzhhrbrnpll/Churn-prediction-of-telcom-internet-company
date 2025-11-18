import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    """Loads the customer churn dataset."""
    try:
        # Coba load file individual dulu
        files = ["Data/Customer_Info.csv", "Data/Location_Data.csv", "Data/Online_Services.csv", "Data/Payment_Info.csv", "Data/Service_Options.csv", "Data/Status_Analysis.csv"]
        dfs = []
        
        for file in files:
            try:
                df_temp = pd.read_csv(file)
                dfs.append(df_temp)
            except Exception as e:
                st.warning(f"File {file} tidak ditemukan: {str(e)}")
        
        if dfs:
            # Gabungkan semua file jika ada multiple files
            if len(dfs) > 1:
                # Gabungkan berdasarkan kolom umum atau gunakan concat sederhana
                df_combined = pd.concat(dfs, axis=1)
                # Hapus kolom duplikat
                df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
                return df_combined
            else:
                return dfs[0]
        else:
            st.error("Tidak ada file yang berhasil dimuat")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# --- DATA PREPROCESSING FOR MODEL ---
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning model following LightGBM approach."""
    df_ml = df.copy()

    # Remove unnecessary columns
    columns_to_drop = [
        'Country', 'state', 'zip_code', 'city', 'latitude', 'longitude',  # Unnecessary location features
        'phone_service_y', 'internet_service_y',  # Duplicate columns
        'Churn_reason', 'churn_category', 'churn_score', 'churn_label', 'churn_status',  # Related features
        'married', 'partner', 'internet_service_x', 'total_revenue', 'total_long_distance_charges', 
        'total_charges', 'total_extra_data_charges', 'total_refunds', 'internet_type_No.'  # Multicollinearity
    ]
    
    # Only drop columns that exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_ml.columns]
    df_ml = df_ml.drop(existing_columns_to_drop, axis=1)
    
    # Remove CustomerID if exists
    if 'customer_id' in df_ml.columns:
        df_ml = df_ml.drop('customer_id', axis=1)

    # Handle missing values
    numeric_columns = df_ml.select_dtypes(include=[np.number]).columns
    categorical_columns = df_ml.select_dtypes(include=['object']).columns

    # Fill numerical missing values with median
    for col in numeric_columns:
        if df_ml[col].isnull().sum() > 0:
            df_ml[col].fillna(df_ml[col].median(), inplace=True)

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if df_ml[col].isnull().sum() > 0:
            df_ml[col].fillna(df_ml[col].mode()[0], inplace=True)

    # Encode categorical variables using LabelEncoder
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le

    return df_ml, label_encoders

# --- MODEL TRAINING WITH LIGHTGBM AND SMOTE ---
@st.cache_resource
def train_lightgbm_model(df_ml):
    """Train LightGBM model with hyperparameter tuning and SMOTE."""

    # Prepare features and target - using churn_value as target variable
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
        'max_depth': [3, 5, 7, -1],  # -1 means no limit
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

# Create tabs for EDA and Prediction
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🤖 Churn Prediction Model", "🔍 Data Overview"])

with tab1:
    # --- EDA CONTENT ---
    st.markdown("""
    This application performs exploratory data analysis (EDA) on the Customer Churn dataset.
    Use the filters in the sidebar to explore customer behavior and churn patterns.
    """)

    # Display info about columns that will be dropped
    with st.expander("⚠️ Data Preprocessing Information"):
        st.info("""
        **Columns that will be automatically removed during model training:**
        - Unnecessary Features: Country, state, zip_code, city, latitude, longitude
        - Duplicate columns: phone_service_y, internet_service_y  
        - Related Features: Churn_reason, churn_category, churn_score, churn_label, churn_status
        - Multicollinearity: married, partner, internet_service_x, total_revenue, total_long_distance_charges, total_charges, total_extra_data_charges, total_refunds, internet_type_No.
        - Target variable: churn_value
        """)

    # --- SIDEBAR FOR FILTERS ---
    st.sidebar.header("Filter Customers")

    # Filter for Churn status - using churn_value
    if 'churn_value' in df.columns:
        churn_status = st.sidebar.multiselect(
            "Select Churn Status",
            options=df["churn_value"].unique(),
            default=df["churn_value"].unique()
    )
    else:
        churn_status = []

    # Filter for other available columns
    available_columns = df.columns.tolist()
    
    # Dynamic filters for other categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove columns that will be dropped from filters
    excluded_from_filters = ['Country', 'state', 'zip_code', 'city', 'latitude', 'longitude', 
                           'phone_service_y', 'internet_service_y', 'Churn_reason', 'churn_category', 
                           'churn_score', 'churn_label', 'churn_status', 'married', 'partner', 
                           'internet_service_x', 'total_revenue', 'total_long_distance_charges', 
                           'total_charges', 'total_extra_data_charges', 'total_refunds', 'internet_type_No.']
    
    filterable_cols = [col for col in categorical_cols if col not in excluded_from_filters]
    
    # Create filters for remaining categorical columns (limit to 5 to avoid too many filters)
    for col in filterable_cols[:5]:
        if col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) <= 20:  # Only create filter if reasonable number of categories
                selected = st.sidebar.multiselect(
                    f"Select {col}",
                    options=unique_vals,
                    default=unique_vals
                )

    # Filter for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['churn_value'] + excluded_from_filters]
    
    for col in numerical_cols[:3]:  # Limit to 3 numerical filters
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
    if churn_status and 'churn_value' in df_selection.columns:
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
            if 'churn_value' in df_selection.columns:
                churn_rate = (df_selection["churn_value"].sum() / total_customers) * 100
                st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")
            else:
                st.metric(label="Dataset Rows", value=total_customers)

        with col3:
            if 'churn_value' in df_selection.columns:
                st.metric(label="Churned Customers", value=df_selection["churn_value"].sum())

        # Additional metrics for numerical columns
        col4, col5, col6 = st.columns(3)

        # Find some numerical columns to display
        numerical_cols_for_display = [col for col in numerical_cols if col in df_selection.columns][:3]
        
        for i, col in enumerate(numerical_cols_for_display):
            with [col4, col5, col6][i]:
                avg_value = round(df_selection[col].mean(), 1)
                st.metric(label=f"Avg. {col}", value=avg_value)

        st.markdown("---")

        # --- VISUALIZATIONS ---
        st.subheader("📊 Visualizations")

        # Arrange charts in columns
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Churn distribution
            if 'churn_value' in df_selection.columns:
                st.subheader("Churn Distribution")
                churn_counts = df_selection["churn_value"].value_counts()
                churn_df = pd.DataFrame({
                    'Churn Status': ['Not Churned', 'Churned'],
                    'Count': churn_counts.values
                })
                st.bar_chart(churn_df.set_index('Churn Status'))

        with viz_col2:
            # Try to show relationship between a numerical feature and churn
            if 'churn_value' in df_selection.columns and len(numerical_cols_for_display) > 0:
                numerical_feature = numerical_cols_for_display[0]
                st.subheader(f"Avg. {numerical_feature} by Churn Status")
                avg_by_churn = df_selection.groupby('churn_value')[numerical_feature].mean()
                st.bar_chart(avg_by_churn)

        # Second row of visualizations
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # Show distribution of a categorical variable
            if len(filterable_cols) > 0:
                cat_feature = filterable_cols[0]
                if cat_feature in df_selection.columns:
                    st.subheader(f"{cat_feature} Distribution")
                    cat_counts = df_selection[cat_feature].value_counts()
                    st.bar_chart(cat_counts)

        with viz_col4:
            # Show distribution of a numerical variable
            if len(numerical_cols_for_display) > 1:
                num_feature = numerical_cols_for_display[1]
                st.subheader(f"{num_feature} Distribution")
                hist_data = df_selection[num_feature].value_counts().sort_index()
                st.line_chart(hist_data)

        # --- DATA SUMMARY ---
        st.subheader("📋 Data Summary")

        # Display some statistics
        col9, col10 = st.columns(2)

        with col9:
            st.write("**Churn Statistics:**")
            if 'churn_value' in df_selection.columns:
                st.write(f"- Churned Customers: {df_selection['churn_value'].sum()}")
                st.write(f"- Non-Churned Customers: {len(df_selection) - df_selection['churn_value'].sum()}")
                st.write(f"- Churn Rate: {churn_rate:.2f}%")

        with col10:
            st.write("**Dataset Information:**")
            st.write(f"- Total Features: {df_selection.shape[1]}")
            st.write(f"- Total Samples: {df_selection.shape[0]}")
            st.write(f"- Numerical Features: {len(df_selection.select_dtypes(include=[np.number]).columns)}")
            st.write(f"- Categorical Features: {len(df_selection.select_dtypes(include=['object']).columns)}")

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

    if 'churn_value' not in df.columns:
        st.error("Target variable 'churn_value' not found in dataset. Please check your data.")
        st.stop()

    # Display columns that will be dropped
    with st.expander("🔧 Data Preprocessing Steps"):
        st.write("**Columns that will be removed:**")
        columns_to_drop = [
            'Country', 'state', 'zip_code', 'city', 'latitude', 'longitude',
            'phone_service_y', 'internet_service_y',
            'Churn_reason', 'churn_category', 'churn_score', 'churn_label', 'churn_status',
            'married', 'partner', 'internet_service_x', 'total_revenue', 'total_long_distance_charges',
            'total_charges', 'total_extra_data_charges', 'total_refunds', 'internet_type_No.'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        st.write(existing_columns_to_drop)
        
        st.write("**Remaining features for modeling:**")
        remaining_cols = [col for col in df.columns if col not in existing_columns_to_drop + ['churn_value']]
        st.write(remaining_cols)

    # Preprocess data for ML
    df_ml, label_encoders = preprocess_data(df)

    # Display preprocessing info
    with st.expander("Data Preprocessing Details"):
        st.write("**Original Data Shape:**", df.shape)
        st.write("**Processed Data Shape:**", df_ml.shape)
        st.write("**Categorical Columns Encoded:**", list(label_encoders.keys()))
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
                if df[feature].dtype in ['int64', 'float64']:
                    # Numerical features
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                else:
                    # Categorical features
                    unique_vals = df[feature].unique()
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals
                    )

        with col2:
            for feature in feature_columns[len(feature_columns)//2:]:
                if df[feature].dtype in ['int64', 'float64']:
                    # Numerical features
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                else:
                    # Categorical features
                    unique_vals = df[feature].unique()
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_vals
                    )

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            for col in input_df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen labels
                    if input_df[col].iloc[0] in le.classes_:
                        input_df[col] = le.transform(input_df[col])
                    else:
                        # If unseen, use the most frequent class
                        input_df[col] = le.transform([le.classes_[0]])

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

            # Explanation
            with st.expander("Understanding the Prediction"):
                st.write("""
                - **Churn Probability**: The model's confidence that the customer will churn
                - **Confidence**: The higher value between churn and non-churn probability
                - **Recommendation**:
                  - For high-risk customers (>50% probability): Implement immediate retention strategies
                  - For low-risk customers (<50% probability): Continue with standard engagement
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


with tab3:
    # --- DATA OVERVIEW TAB ---
    st.header("🔍 Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data types
        st.write("**Data Types:**")
        dtype_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_info)
        
        # Show columns that will be dropped
        st.write("**Columns to be removed for modeling:**")
        columns_to_drop = [
            'Country', 'state', 'zip_code', 'city', 'latitude', 'longitude',
            'phone_service_y', 'internet_service_y',
            'Churn_reason', 'churn_category', 'churn_score', 'churn_label', 'churn_status',
            'married', 'partner', 'internet_service_x', 'total_revenue', 'total_long_distance_charges',
            'total_charges', 'total_extra_data_charges', 'total_refunds', 'internet_type_No.'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        st.write(existing_columns_to_drop)

    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

    # Missing values
    st.subheader("🔍 Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percent
    })
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])

    # Correlation matrix (if numerical columns exist)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        st.subheader("📊 Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()

        if matplotlib_available:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.dataframe(corr_matrix)

    # --- DISPLAY RAW DATA ---
    with st.expander("View Raw Data"):
        st.dataframe(df)
        st.markdown(f"**Data Dimensions:** {df.shape[0]} rows, {df.shape[1]} columns")

st.markdown("---")
st.write("Customer Churn Analysis & Prediction Dashboard | Model: LightGBM with SMOTE")
