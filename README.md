# Telco-Churn-Prediction
## Project Overview
 - Goal: Predict whether a customer will churn (leave the service) based on demographic and
 service usage data.
 - Dataset: Telco Customer Churn Dataset (IBM Sample Data)
 - Approach:
 1. Exploratory Data Analysis (EDA)
 2. Data cleaning & preprocessing
 3. Model training & evaluation
 4. Deployment with a Streamlit dashboard

## Repository Structure
 Telco-Churn-Prediction/
 1. app.py # Streamlit app
 2. Telco_Churn_Prediction.ipynb # Jupyter notebook with EDA & model training
 3. churn_pipeline.pkl # Pre-trained model pipeline (optional)
 4. requirements.txt # Dependencies
 5. README.md # Project documentation

## Dataset Description
 - Rows: ~7,043 customers
 - Columns:
  • Demographics: gender, SeniorCitizen, Partner, Dependents
  • Services: PhoneService, InternetService, OnlineSecurity, etc.
  • Account Info: Contract type, PaymentMethod, PaperlessBilling
  • Financials: MonthlyCharges, TotalCharges
  • Target: Churn (Yes/No)

## Methodology
 1. Data Preprocessing
  - Handled missing values (esp. TotalCharges)
  - Converted categorical variables using OneHotEncoding
  - Normalized numerical variables (MonthlyCharges, TotalCharges, tenure)
  - Dealt with class imbalance using stratified splits

 2. Exploratory Data Analysis (EDA)
  - Churn Rate: ~26.5% customers churned
  - Key Findings:
   • Month-to-month contracts → highest churn
   • Fiber optic internet users churn more
   • Senior citizens and high monthly charges = higher risk
   • Longer tenure customers churn less

 3. Modeling
  - Models Tested: Logistic Regression, Random Forest Classifier
  - Evaluation Metric: ROC-AUC Score
  - In the notebook: Logistic Regression performed best (ROC-AUC ≈ 0.84)
  - In the Streamlit app: Both models are trained and compared; whichever achieves the best ROC-AUC is selected automatically.

## Visualizations
  - Monthly Charges vs Churn: Customers with high monthly charges are more likely to churn.
  - Tenure Distribution: New customers (low tenure) churn more often.
  - Contract Type vs Churn:
   • Month-to-month contracts → high churn
   • Two-year contracts → very low churn
   (All plots included in the Jupyter notebook.)

## Streamlit Dashboard
  The app allows predictions in two modes:
  1. Single Prediction
   - Enter customer details manually
   - Get prediction + probability
  2. Batch Prediction
   - Upload CSV with multiple customers
   - Get predictions + downloadable CSV
  Features:
   - Handles missing values & unseen categories
   - Repair mode: retrain pipeline directly in app if pickle fails
   - Dynamic model selection: automatically picks best between Logistic Regression and Random Forest.
 
 ## Installation & Usage
   Clone Repository:
   git clone https://github.com/yourusername/Telco-Churn-Prediction.git
   cd Telco-Churn-Prediction
 
   Install Requirements:
   pip install -r requirements.txt
 
   Run App:
    streamlit run app.py
   
   Requirements
   - Python 3.8+
   - Libraries: pandas, numpy, scikit-learn, streamlit, joblib, matplotlib, seaborn
  
 ## Results
   - In Notebook:
    • Best model: Logistic Regression
    • ROC-AUC Score: ~0.84
   - In Streamlit App:
    • Best model is chosen dynamically (Logistic Regression or Random Forest)
    • Ensures the most robust pipeline depending on environment and data split
   - Key Drivers of Churn:
    • Month-to-month contract
    • High monthly charges
    • Lack of additional services (e.g., OnlineSecurity, TechSupport)
    • Senior citizen status
 
 ## Key Learnings
   - Data preprocessing is critical.
   - Feature importance reveals business drivers of churn.
   - Logistic Regression provided strong, interpretable results.
   - Streamlit makes the model accessible to non-technical stakeholders.
 
