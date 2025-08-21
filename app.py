import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sklearn

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📞", layout="wide")

# =============================
# Version banner (helps diagnose pickle mismatches)
# =============================
st.sidebar.info(f"scikit-learn version: **{sklearn.__version__}**")

# =============================
# Expected schema from the Telco dataset
# =============================
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [c for c in REQUIRED_COLUMNS if c not in NUMERIC_COLS]

# Defaults for categorical imputations
DEFAULT_CATS = {
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",  # "No", "Yes", "No phone service"
    "InternetService": "DSL",  # "DSL", "Fiber optic", "No"
    "OnlineSecurity": "No",  # "No", "Yes", "No internet service"
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

# =============================
# Data cleaning for inference (single & batch)
# =============================

def coerce_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Churn", "customerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "SeniorCitizen" in df.columns:
        mapping = {"Yes": 1, "No": 0, "Y": 1, "N": 0, True: 1, False: 0}
        df["SeniorCitizen"] = df["SeniorCitizen"].map(mapping).fillna(df["SeniorCitizen"])
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        df["SeniorCitizen"] = df["SeniorCitizen"].clip(lower=0, upper=1)

    for col in NUMERIC_COLS:
        median_val = df[col].median() if df[col].notna().any() else 0.0
        df[col] = df[col].fillna(median_val)

    def clean_cat(col: str, allowed: list):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        default_val = DEFAULT_CATS.get(col, allowed[0])
        df[col] = df[col].fillna(default_val)
        df[col] = df[col].where(df[col].isin(allowed), default_val)

    clean_cat("gender", ["Male", "Female"])
    clean_cat("Partner", ["Yes", "No"])
    clean_cat("Dependents", ["Yes", "No"])
    clean_cat("PhoneService", ["Yes", "No"])
    clean_cat("MultipleLines", ["No", "Yes", "No phone service"])
    clean_cat("InternetService", ["DSL", "Fiber optic", "No"])
    clean_cat("OnlineSecurity", ["No", "Yes", "No internet service"])
    clean_cat("OnlineBackup", ["No", "Yes", "No internet service"])
    clean_cat("DeviceProtection", ["No", "Yes", "No internet service"])
    clean_cat("TechSupport", ["No", "Yes", "No internet service"])
    clean_cat("StreamingTV", ["No", "Yes", "No internet service"])
    clean_cat("StreamingMovies", ["No", "Yes", "No internet service"])
    clean_cat("Contract", ["Month-to-month", "One year", "Two year"])
    clean_cat("PaperlessBilling", ["Yes", "No"])
    clean_cat("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    df["tenure"] = df["tenure"].clip(lower=0, upper=100)
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0)
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)

    return df

# =============================
# Build a fresh pipeline (current environment) — used as fallback when pickle can't load
# =============================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="drop",  # avoid version-specific remainder internals
        verbose_feature_names_out=False,
    )


def train_fallback_pipeline(train_df: pd.DataFrame):
    train_df = train_df.copy()
    # Clean training data like in your notebook
    if "customerID" in train_df.columns:
        train_df = train_df.drop("customerID", axis=1)
    train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
    train_df["TotalCharges"] = train_df["TotalCharges"].fillna(train_df["TotalCharges"].median())

    y = train_df["Churn"].map({"No": 0, "Yes": 1})
    X = train_df.drop("Churn", axis=1)

    X = coerce_and_impute(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = build_preprocessor()

    candidates = [
        ("LogReg", Pipeline([("preprocessor", pre), ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42))])),
        ("RF", Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(n_estimators=300, random_state=42))]))
    ]

    best_name, best_pipe, best_auc = None, None, -1
    for name, pipe in candidates:
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob)
        if auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, auc

    return best_pipe, {"model": best_name, "roc_auc": best_auc}

# =============================
# Try to load existing pipeline; if it fails, allow user to re-train inside the app
# =============================

@st.cache_resource(show_spinner=False)
def safe_load_pipeline(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        return e  # return the exception for handling

pipe_or_exc = safe_load_pipeline("churn_pipeline.pkl")

if isinstance(pipe_or_exc, Exception):
    st.error(f"❌ Could not load `churn_pipeline.pkl` — {pipe_or_exc}")
    with st.expander("Repair mode: Re-train pipeline in this environment", expanded=True):
        st.write("Upload the original **Telco training CSV** (WA_Fn-UseC_-Telco-Customer-Churn.csv). We'll rebuild and resave the pipeline using your current scikit-learn version.")
        train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train_csv")
        if train_file is not None:
            try:
                train_df = pd.read_csv(train_file)
                with st.spinner("Training pipeline (this usually takes < 1 minute)…"):
                    pipeline, meta = train_fallback_pipeline(train_df)
                joblib.dump(pipeline, "churn_pipeline.pkl")
                st.success(f"✅ Re-trained and saved pipeline (model={meta['model']}, AUC={meta['roc_auc']:.3f}). You can now use predictions below.")
            except Exception as e:
                st.error(f"Training failed: {e}")
        else:
            st.info("No training file uploaded yet.")
else:
    pipeline = pipe_or_exc

st.title("📞 Telco Customer Churn Prediction — Single & Batch (Robust)")
st.caption("Handles missing values, normalizes categoricals, and avoids feature-mismatch. If your saved pipeline couldn't load, use Repair mode above to rebuild under this scikit-learn version.")

# =============================
# CSV Template
# =============================
@st.cache_data(show_spinner=False)
def csv_template_bytes() -> bytes:
    template = pd.DataFrame([{c: (0 if c in NUMERIC_COLS or c=="SeniorCitizen" else "") for c in REQUIRED_COLUMNS}])
    template.loc[0] = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 50.0
    }
    return template.to_csv(index=False).encode("utf-8")

with st.sidebar:
    st.subheader("📄 Template")
    st.download_button(
        label="Download CSV Template",
        data=csv_template_bytes(),
        file_name="telco_churn_template.csv",
        mime="text/csv"
    )

# =============================
# Mode selection (Single vs Batch)
# =============================
mode = st.radio("Select prediction mode:", ["Single Prediction", "Batch CSV Prediction"], horizontal=True)

# =============================
# SINGLE PREDICTION UI
# =============================
if mode == "Single Prediction":
    st.header("🔹 Single Customer Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=50.0)

    if st.button("Predict Churn", type="primary"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        clean_df = coerce_and_impute(input_df)
        try:
            proba = float(pipeline.predict_proba(clean_df)[:, 1][0])
            pred = int(pipeline.predict(clean_df)[0])
            st.success(f"Prediction: {'Churn' if pred==1 else 'No Churn'}  |  Probability: {proba:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# =============================
# BATCH CSV PREDICTION UI
# =============================
else:
    st.header("📂 Batch CSV Prediction")
    st.write("Upload a CSV of one or more customers. We'll validate, impute, and predict.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("### Preview (first 10 rows)")
            st.dataframe(raw_df.head(10))

            clean_df = coerce_and_impute(raw_df)
            preds = pipeline.predict(clean_df)
            probas = pipeline.predict_proba(clean_df)[:, 1]

            results_df = raw_df.copy()
            results_df["Churn_Prediction"] = np.where(preds==1, "Yes", "No")
            results_df["Churn_Probability"] = probas

            st.write("### Results")
            st.dataframe(results_df)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_bytes,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to start batch prediction.")

# Footer
st.caption("If loading a saved pickle fails due to scikit-learn changes, use the Repair mode above to rebuild the pipeline under the current version. To avoid future issues, pin scikit-learn to the same version used for training.")
