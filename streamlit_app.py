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
        files = ["Customer_Info.csv", "Location_Data.csv", "Online_Services.csv", "Payment_Info.csv", "Service_Options.csv", "Status_Analysis.csv"]
        dfs = []
        
        for file in files:
            try:
                df_temp = pd.read_csv(file)
                dfs.append(df_temp)
                st.success(f"Berhasil memuat: {file}")
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
        

df = load_data()

# --- DATA PREPROCESSING FOR MODEL ---
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning model following LightGBM approach."""
    df_ml = df.copy()

    # Remove CustomerID if exists
    if 'CustomerID' in df_ml.columns:
        df_ml = df_ml.drop('CustomerID', axis=1)

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

    # Prepare features and target
    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']

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

    # --- SIDEBAR FOR FILTERS ---
    st.sidebar.header("Filter Customers")

    # Filter for Churn status
    if 'Churn' in df.columns:
        churn_status = st.sidebar.multiselect(
            "Select Churn Status",
            options=df["Churn"].unique(),
            default=df["Churn"].unique()
        )
    else:
        churn_status = []

    # Filter for City Tier (if exists)
    if 'CityTier' in df.columns:
        city_tier = st.sidebar.multiselect(
            "Select City Tier",
            options=df["CityTier"].unique(),
            default=df["CityTier"].unique()
        )
    else:
        city_tier = []

    # Filter for Gender (if exists)
    if 'Gender' in df.columns:
        gender = st.sidebar.multiselect(
            "Select Gender",
            options=df["Gender"].unique(),
            default=df["Gender"].unique()
        )
    else:
        gender = []

    # Filter for Preferred Login Device (if exists)
    if 'PreferredLoginDevice' in df.columns:
        login_device = st.sidebar.multiselect(
            "Select Preferred Login Device",
            options=df["PreferredLoginDevice"].unique(),
            default=df["PreferredLoginDevice"].unique()
        )
    else:
        login_device = []

    # Filter for Preferred Payment Mode (if exists)
    if 'PreferredPaymentMode' in df.columns:
        payment_mode = st.sidebar.multiselect(
            "Select Preferred Payment Mode",
            options=df["PreferredPaymentMode"].unique(),
            default=df["PreferredPaymentMode"].unique()
        )
    else:
        payment_mode = []

    # Filter for Preferred Order Category (if exists)
    if 'PreferedOrderCat' in df.columns:
        order_cat = st.sidebar.multiselect(
            "Select Preferred Order Category",
            options=df["PreferedOrderCat"].unique(),
            default=df["PreferedOrderCat"].unique()
        )
    else:
        order_cat = []

    # Filter for Tenure range (if exists)
    if 'Tenure' in df.columns:
        min_tenure, max_tenure = int(df["Tenure"].min()), int(df["Tenure"].max())
        tenure_slider = st.sidebar.slider(
            "Select Tenure Range",
            min_value=min_tenure,
            max_value=max_tenure,
            value=(min_tenure, max_tenure),
        )
    else:
        tenure_slider = (0, 1)

    # Filter for Satisfaction Score (if exists)
    if 'SatisfactionScore' in df.columns:
        satisfaction = st.sidebar.multiselect(
            "Select Satisfaction Score",
            options=df["SatisfactionScore"].unique(),
            default=df["SatisfactionScore"].unique()
        )
    else:
        satisfaction = []

    # --- FILTERING THE DATAFRAME ---
    df_selection = df.copy()

    # Apply multiselect filters only if a selection has been made for that filter
    if churn_status and 'Churn' in df_selection.columns:
        df_selection = df_selection[df_selection["Churn"].isin(churn_status)]
    if city_tier and 'CityTier' in df_selection.columns:
        df_selection = df_selection[df_selection["CityTier"].isin(city_tier)]
    if gender and 'Gender' in df_selection.columns:
        df_selection = df_selection[df_selection["Gender"].isin(gender)]
    if login_device and 'PreferredLoginDevice' in df_selection.columns:
        df_selection = df_selection[df_selection["PreferredLoginDevice"].isin(login_device)]
    if payment_mode and 'PreferredPaymentMode' in df_selection.columns:
        df_selection = df_selection[df_selection["PreferredPaymentMode"].isin(payment_mode)]
    if order_cat and 'PreferedOrderCat' in df_selection.columns:
        df_selection = df_selection[df_selection["PreferedOrderCat"].isin(order_cat)]
    if satisfaction and 'SatisfactionScore' in df_selection.columns:
        df_selection = df_selection[df_selection["SatisfactionScore"].isin(satisfaction)]

    # Apply slider filter
    if 'Tenure' in df_selection.columns:
        df_selection = df_selection[
            (df_selection["Tenure"] >= tenure_slider[0]) &
            (df_selection["Tenure"] <= tenure_slider[1])
        ]

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
            if 'Churn' in df_selection.columns:
                churn_rate = (df_selection["Churn"].sum() / total_customers) * 100
                st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")
            else:
                st.metric(label="Dataset Rows", value=total_customers)

        with col3:
            if 'Tenure' in df_selection.columns:
                avg_tenure = round(df_selection["Tenure"].mean(), 1)
                st.metric(label="Average Tenure", value=avg_tenure)
            elif 'Churn' in df_selection.columns:
                st.metric(label="Churned Customers", value=df_selection["Churn"].sum())

        # Additional metrics
        col4, col5, col6 = st.columns(3)

        with col4:
            if 'OrderCount' in df_selection.columns:
                avg_order_count = round(df_selection["OrderCount"].mean(), 1)
                st.metric(label="Avg. Order Count", value=avg_order_count)
            else:
                if 'Complain' in df_selection.columns:
                    complain_rate = (df_selection["Complain"].sum() / total_customers) * 100
                    st.metric(label="Complain Rate", value=f"{complain_rate:.1f}%")

        with col5:
            if 'CashbackAmount' in df_selection.columns:
                avg_cashback = round(df_selection["CashbackAmount"].mean(), 1)
                st.metric(label="Avg. Cashback", value=avg_cashback)
            elif 'Churn' in df_selection.columns:
                st.metric(label="Non-Churned Customers", value=len(df_selection) - df_selection["Churn"].sum())

        with col6:
            if 'HourSpendOnApp' in df_selection.columns:
                avg_hours_app = round(df_selection["HourSpendOnApp"].mean(), 1)
                st.metric(label="Avg. Hours on App", value=avg_hours_app)
            elif 'NumberOfDeviceRegistered' in df_selection.columns:
                avg_devices = round(df_selection["NumberOfDeviceRegistered"].mean(), 1)
                st.metric(label="Avg. Devices Registered", value=avg_devices)

        st.markdown("---")

        # --- VISUALIZATIONS ---
        st.subheader("📊 Visualizations")

        # Arrange charts in columns
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Churn distribution
            if 'Churn' in df_selection.columns:
                st.subheader("Churn Distribution")
                churn_counts = df_selection["Churn"].value_counts()
                churn_df = pd.DataFrame({
                    'Churn Status': ['Not Churned', 'Churned'],
                    'Count': churn_counts.values
                })
                st.bar_chart(churn_df.set_index('Churn Status'))

        with viz_col2:
            # Try different columns for second chart
            if 'CashbackAmount' in df_selection.columns:
                # Average Cashback by Churn Status
                st.subheader("Avg. Cashback by Churn Status")
                if 'Churn' in df_selection.columns:
                    avg_cashback_by_churn = df_selection.groupby('Churn')['CashbackAmount'].mean()
                    st.bar_chart(avg_cashback_by_churn)
            elif 'Tenure' in df_selection.columns and 'Churn' in df_selection.columns:
                # Tenure by Churn Status
                st.subheader("Avg. Tenure by Churn Status")
                avg_tenure_by_churn = df_selection.groupby('Churn')['Tenure'].mean()
                st.bar_chart(avg_tenure_by_churn)

        # Second row of visualizations
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # Satisfaction Score Distribution if available
            if 'SatisfactionScore' in df_selection.columns:
                st.subheader("Satisfaction Score Distribution")
                satisfaction_counts = df_selection["SatisfactionScore"].value_counts().sort_index()
                st.bar_chart(satisfaction_counts)
            elif 'OrderCount' in df_selection.columns:
                st.subheader("Order Count Distribution")
                order_counts = df_selection["OrderCount"].value_counts().sort_index()
                st.bar_chart(order_counts.head(20))  # Limit to top 20

        with viz_col4:
            # Tenure Distribution if available
            if 'Tenure' in df_selection.columns:
                st.subheader("Tenure Distribution")
                tenure_hist = df_selection["Tenure"].value_counts().sort_index()
                st.line_chart(tenure_hist)
            elif 'HourSpendOnApp' in df_selection.columns:
                st.subheader("Hours Spent on App")
                hours_hist = df_selection["HourSpendOnApp"].value_counts().sort_index()
                st.line_chart(hours_hist)

        # Third row of visualizations (if categorical columns exist)
        viz_col5, viz_col6 = st.columns(2)

        with viz_col5:
            if 'PreferredPaymentMode' in df_selection.columns:
                # Preferred Payment Mode
                st.subheader("Preferred Payment Mode")
                payment_counts = df_selection["PreferredPaymentMode"].value_counts()
                st.bar_chart(payment_counts)
            elif 'Gender' in df_selection.columns:
                # Gender Distribution
                st.subheader("Gender Distribution")
                gender_counts = df_selection["Gender"].value_counts()
                st.bar_chart(gender_counts)

        with viz_col6:
            if 'CityTier' in df_selection.columns:
                # City Tier Distribution
                st.subheader("City Tier Distribution")
                city_counts = df_selection["CityTier"].value_counts().sort_index()
                st.bar_chart(city_counts)
            elif 'PreferedOrderCat' in df_selection.columns:
                # Preferred Order Category
                st.subheader("Preferred Order Category")
                order_cat_counts = df_selection["PreferedOrderCat"].value_counts()
                st.bar_chart(order_cat_counts)

        # --- DATA SUMMARY ---
        st.subheader("📋 Data Summary")

        # Display some statistics
        col9, col10 = st.columns(2)

        with col9:
            st.write("**Churn Statistics:**")
            if 'Churn' in df_selection.columns:
                st.write(f"- Churned Customers: {df_selection['Churn'].sum()}")
                st.write(f"- Non-Churned Customers: {len(df_selection) - df_selection['Churn'].sum()}")
                st.write(f"- Churn Rate: {churn_rate:.2f}%")

        with col10:
            st.write("**Engagement Statistics:**")
            if 'HourSpendOnApp' in df_selection.columns:
                st.write(f"- Avg Hours on App: {avg_hours_app}")
            if 'NumberOfDeviceRegistered' in df_selection.columns:
                st.write(f"- Avg Devices Registered: {df_selection['NumberOfDeviceRegistered'].mean():.1f}")
            if 'Complain' in df_selection.columns:
                st.write(f"- Customers with Complaints: {df_selection['Complain'].sum()}")

        # --- DISPLAY RAW DATA ---
        with st.expander("View Filtered Data"):
            st.dataframe(df_selection)
            st.markdown(f"**Data Dimensions:** {df_selection.shape[0]} rows, {df_selection.shape[1]} columns")

with tab2:
    # --- CHURN PREDICTION MODEL ---
    st.header("🤖 Customer Churn Prediction Model")
    st.markdown("""
    This section uses a tuned LightGBM classifier with SMOTE to handle class imbalance for customer churn prediction.
    """)

    if 'Churn' not in df.columns:
        st.error("Target variable 'Churn' not found in dataset. Please check your data.")
        st.stop()

    # Preprocess data for ML
    df_ml, label_encoders = preprocess_data(df)

    # Display preprocessing info
    with st.expander("Data Preprocessing Details"):
        st.write("**Original Data Shape:**", df.shape)
        st.write("**Processed Data Shape:**", df_ml.shape)
        st.write("**Categorical Columns Encoded:**", list(label_encoders.keys()))
        st.write("**Class Distribution:**")
        st.write(df_ml['Churn'].value_counts())

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
                        min_val, max_val, default_val
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
                        min_val, max_val, default_val
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
