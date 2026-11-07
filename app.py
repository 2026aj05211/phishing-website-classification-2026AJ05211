# ============================================================
# app.py — Phishing Website Classifier
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
    page_title="Phishing Website Classifier — 2026AJ05211",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.0rem; font-weight: 700; color: #1f4e79;
            text-align: center; margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 0.95rem; color: #666; text-align: center;
            margin-bottom: 1.2rem;
        }
        .section-header {
            font-size: 1.2rem; font-weight: 600; color: #1f4e79;
            margin-top: 1.4rem; margin-bottom: 0.6rem;
            border-bottom: 2px solid #1f4e79; padding-bottom: 0.2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem; color: #1f4e79;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Constants ----------------
# Standard feature order for UCI Phishing Websites dataset (ID 327).
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

# ---------------- Main ----------------
def main():
    st.markdown('<div class="main-title">🛡️ Phishing Website Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Machine Learning Assignment 2 &nbsp;|&nbsp; BITS ID: 2026AJ05211</div>',
        unsafe_allow_html=True,
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**1. Upload test data (CSV)**")
        uploaded_file = st.file_uploader(
            "Upload test_data.csv", type=["csv"], label_visibility="collapsed"
        )

        st.markdown("**2. Select classifier**")
        selected_model_name = st.selectbox(
            "Choose a model",
            options=list(MODEL_FILES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Dataset**")
        st.markdown(
            "- UCI Phishing Websites (ID 327)\n"
            "- 30 features, 2,456 instances\n"
            "- Binary classification\n"
            "- 0 = Phishing, 1 = Legitimate"
        )
        st.markdown("---")
        st.caption("Models loaded from disk. No retraining is performed.")

    # ---------- No file yet ----------
    if uploaded_file is None:
        st.info("👈 Upload **test_data.csv** from the sidebar to begin.")
        st.markdown("### How to use")
        st.markdown(
            "1. Upload the `test_data.csv` file (30 features + `Result` column).\n"
            "2. Select one of the 5 classifiers from the sidebar.\n"
            "3. The app loads the saved model and computes evaluation metrics — **no retraining**."
        )
        return

    # ---------- Read uploaded CSV ----------
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return

    st.markdown('<div class="section-header">📄 Uploaded Test Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------- Validate ----------
    if TARGET_COLUMN not in df.columns:
        st.error(f"❌ Uploaded CSV must contain a '{TARGET_COLUMN}' column.")
        return

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required feature columns: {missing}")
        return

    # ---------- Prepare & predict ----------
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

    # ---------- Metrics ----------
    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.markdown(
        f'<div class="section-header">📊 Evaluation Metrics — {selected_model_name}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC Score", f"{metrics['AUC']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall",   f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1']:.4f}")
    c6.metric("MCC Score", f"{metrics['MCC']:.4f}")

    # ---------- Confusion matrix + classification report ----------
    st.markdown('<div class="section-header">🔢 Confusion Matrix & Classification Report</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_true, y_pred)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Phishing (0)", "Legitimate (1)"],
            yticklabels=["Phishing (0)", "Legitimate (1)"],
            ax=ax, cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model_name, fontsize=11)
        st.pyplot(fig)

    with col_b:
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["Phishing (0)", "Legitimate (1)"],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.markdown("**Classification Report**")
        st.dataframe(report_df, use_container_width=True)

    # ---------- Prediction sample ----------
    st.markdown('<div class="section-header">🔍 Prediction Sample (first 20 rows)</div>', unsafe_allow_html=True)
    result_df = pd.DataFrame({
        "Actual (Result)": y_true,
        "Predicted":       y_pred,
        "Correct":         y_true == y_pred,
    })
    st.dataframe(result_df.head(20), use_container_width=True)

    st.success(
        f"✅ Evaluation complete using **{selected_model_name}** — "
        f"model loaded from disk, **no retraining** performed."
    )

if __name__ == "__main__":
    main()