# ============================================================
# app.py — Phishing Website Classifier (Professional UI)
# BITS WILP M.Tech (AIML) — Machine Learning Assignment 2
# BITS ID: 2026AJ05211
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)
import streamlit as st

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Phishing Classifier | 2026AJ05211",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Custom CSS for Professional Look ----------------
st.markdown(
    """
    <style>
        /* Main background and font */
        .stApp {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #1f4e79;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Header Card */
        .header-card {
            background: linear-gradient(135deg, #1f4e79 0%, #2980b9 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-card h1 {
            color: white !important;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }
        .header-card p {
            font-size: 1.1rem;
            margin: 0;
            opacity: 0.9;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 5px solid #2980b9;
        }
        div[data-testid="stMetricValue"] {
            color: #1f4e79;
            font-weight: bold;
        }
        
        /* Section Headers */
        .section-title {
            color: #1f4e79;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Constants ----------------
FEATURE_COLUMNS = [
    "having_IP_Address", "URL_Length", "Shortining_Service",
    "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix",
    "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
    "Favicon", "port", "HTTPS_token", "Request_URL", "URL_of_Anchor",
    "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow",
    "Iframe", "age_of_domain", "DNSRecord", "web_traffic",
    "Page_Rank", "Google_Index", "Links_pointing_to_page", "Statistical_report",
]
TARGET_COLUMN = "Result"

MODEL_FILES = {
    "Logistic Regression":       "model/logistic_regression.joblib",
    "Decision Tree":             "model/decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "model/knn.joblib",
    "Naive Bayes":               "model/naive_bayes.joblib",
    "Support Vector Machine":    "model/svm.joblib",
}

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load("model/scaler.joblib")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "AUC":       roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }

# ---------------- Main App ----------------
def main():
    # Header
    st.markdown(
        """
        <div class="header-card">
            <h1>🛡️ Phishing Website Classifier</h1>
            <p>Machine Learning Assignment 2 &nbsp;|&nbsp; BITS ID: 2026AJ05211</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1161/1161588.png", width=80)
        st.markdown("### ⚙️ Model Configuration")
        
        st.markdown("**1. Upload Test Data**")
        uploaded_file = st.file_uploader(
            "Upload test_data.csv", type=["csv"], label_visibility="collapsed"
        )

        st.markdown("**2. Select Classifier**")
        selected_model_name = st.selectbox(
            "Choose a model",
            options=list(MODEL_FILES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.info(
            "**Dataset Info**\n"
            "- UCI Phishing Websites (ID 327)\n"
            "- 30 features, 2,456 instances\n"
            "- 0 = Phishing | 1 = Legitimate"
        )
        st.caption("Models are loaded from disk. No retraining is performed.")

    # Main body
    if uploaded_file is None:
        st.info("👈 **Please upload `test_data.csv` from the sidebar to begin.**")
        st.markdown("### How to use this app")
        st.markdown(
            "1. Upload the `test_data.csv` file (must contain 30 features + `Result` column).\n"
            "2. Select one of the 5 trained machine learning classifiers from the sidebar.\n"
            "3. The app will instantly evaluate the model and display performance metrics, a confusion matrix, and a classification report — **without retraining**."
        )
        return

    # Read and validate data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read the uploaded file: {e}")
        return

    if TARGET_COLUMN not in df.columns:
        st.error(f"❌ Uploaded CSV must contain a '{TARGET_COLUMN}' column.")
        return

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required feature columns: {missing}")
        return

    # Display data preview in an expander
    with st.expander("📄 View Uploaded Test Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Prepare data and predict
    X = df[FEATURE_COLUMNS].values
    y_true = df[TARGET_COLUMN].values.astype(int)

    try:
        scaler = load_scaler()
        model = load_model(MODEL_FILES[selected_model_name])
    except FileNotFoundError as e:
        st.error(f"❌ Could not load model/scaler file: {e}")
        return

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_scaled)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # Display Section
    st.markdown(f'<div class="section-title">📊 Evaluation Results: {selected_model_name}</div>', unsafe_allow_html=True)

    # Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC Score", f"{metrics['AUC']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall",    f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score",  f"{metrics['F1']:.4f}")
    c6.metric("MCC Score", f"{metrics['MCC']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for detailed results
    tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix & Report", "🔍 Detailed Predictions", "ℹ️ About this Model"])

    with tab1:
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Phishing (0)", "Legitimate (1)"],
                yticklabels=["Phishing (0)", "Legitimate (1)"],
                ax=ax, cbar=False,
            )
            ax.set_xlabel("Predicted Label", fontweight='bold')
            ax.set_ylabel("True Label", fontweight='bold')
            st.pyplot(fig)

        with col_b:
            st.markdown("**Classification Report**")
            report_dict = classification_report(
                y_true, y_pred,
                target_names=["Phishing (0)", "Legitimate (1)"],
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    with tab2:
        st.markdown("**Prediction Sample (First 20 rows)**")
        result_df = pd.DataFrame({
            "Actual (Result)": y_true,
            "Predicted":       y_pred,
            "Correct":         y_true == y_pred,
        })
        st.dataframe(result_df.head(20), use_container_width=True)

    with tab3:
        st.markdown(f"**Model Information:** {selected_model_name}")
        st.info(
            "This model was trained on the **UCI Phishing Websites dataset** using an 80/20 train-test split. "
            "The features were standardized using `StandardScaler` before training. "
            "The model weights are loaded directly from a saved `.joblib` file to ensure **no retraining occurs on uploaded data**."
        )

    st.success(f"✅ Evaluation complete using **{selected_model_name}** — model loaded from disk, **no retraining** performed.")

if __name__ == "__main__":
    main()
