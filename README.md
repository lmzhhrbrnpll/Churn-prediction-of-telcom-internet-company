# 📱 Telecom Customer Churn Prediction

## Project Overview
This machine learning project focuses on predicting customer churn for a telecommunications company. The model identifies at-risk customers enabling proactive retention strategies and reducing revenue loss.

## 💼 Business Problem
- Customer churn leading to revenue decline and increased acquisition costs
- Difficulty in predicting which customers are likely to churn
- Need for early intervention strategies

## 🎯 Objectives
- Identify main factors causing customer churn
- Build predictive model to classify at-risk customers
- Develop actionable retention strategies

## 📊 Dataset
Multiple data sources were integrated:
- `Customer_Info.csv`, `Location_Data.csv`, `Online_Services.csv`
- `Payment_Info.csv`, `Service_Options.csv`, `Status_Analysis.csv`

### Key Feature Categories:
- **Demographic**: Age, Gender, Senior Citizen, Marital Status
- **Service**: Internet Type, Online Security, Streaming Services
- **Financial**: Contract Type, Payment Method, Monthly Charges
- **Behavioral**: Number of Referrals, Satisfaction Score, Monthly GB Download

## 🔧 Data Preprocessing
- **Data Cleaning**: Removed unnecessary columns, handled missing values
- **Feature Engineering**: Encoding categorical variables, standardization
- **Train-Test Split**: 80-20 split with stratification
- **Handling Imbalance**: SMOTE technique implementation

## 🤖 Model Development & Selection

### Models Evaluated:
- Decision Tree, Random Forest, XGBoost, LightGBM

### Performance Comparison:
| Model | Test F1-Score | ROC-AUC |
|-------|---------------|---------|
| Decision Tree | 88.71% | 91.96% |
| Random Forest | 91.57% | 98.54% |
| XGBoost | 90.91% | 99.08% |
| **LightGBM** | **91.59%** | **99.19%** |

### Final Model: Optimized LightGBM
- **Best Performance**: F1-Score 92.89%, AUC 98.92%
- **Hyperparameter Tuning**: n_estimators, max_depth, learning_rate, etc.
- **Class Balance**: SMOTE implementation

## 📈 Key Findings

### Top 3 Churn Drivers:
1. **Monthly Charges** - Price sensitivity and value perception
2. **Tenure** - Relationship stability with company  
3. **Number of Referrals** - Early warning indicator of loyalty decline

### Model Performance:
- **True Positives**: 1,010 customers correctly identified as churners
- **False Positives**: Only 26 false alarms
- **High AUC (98.92%)**: Excellent at distinguishing churners from non-churners

## 🛠️ Implementation
- **Streamlit App**: Interactive tool for customer churn probability prediction
- **Real-time Scoring**: Business team can input customer details for instant predictions

## 💡 Business Recommendations
1. **Monthly Charge Strategy**: Review pricing structure, offer value-added packages
2. **Referral Program Optimization**: Enhance incentives for customer ambassadors
3. **Family Packages**: Develop comprehensive family bundles for competitive markets

## 🛠️ Technologies Used
- Python (Pandas, Scikit-learn, LightGBM)
- SMOTE for imbalance handling
- Streamlit for deployment
- Hyperparameter optimization

## 📞 Contact
**Lamzahhera Berinpalla**
- LinkedIn: [Lamzahhera Berinpalla](https://www.linkedin.com/in/lamzahheraberinpalla/)
- Email: lamzahheraberin@gmail.com
