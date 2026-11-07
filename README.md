# Phishing Website Classification using Machine Learning

**BITS ID:** 2026AJ05211  
**Course:** M.Tech (AIML) — Machine Learning Assignment 2  
**Institution:** BITS Pilani — WILP Division

---

## a. Problem Statement
Phishing attacks are a major cybersecurity threat where attackers create fraudulent websites to steal sensitive information such as login credentials, credit-card numbers, and personal data. This project builds and compares **five machine learning classification models** that distinguish legitimate websites from phishing websites using **30 URL and webpage features** extracted from the UCI Phishing Websites dataset. The trained models are deployed in an interactive Streamlit application that evaluates uploaded test data without retraining.

---

## b. Dataset Description
- **Source:** UCI Machine Learning Repository — Phishing Websites Dataset (ID 327)
- **Instances:** 2,456
- **Features:** 30 (all numeric, integer-encoded: −1 / 0 / 1)
- **Target:** Binary — `Result` (1 = legitimate, 0 = phishing)
- **Subject Area:** Computer Science / Cybersecurity
- **Missing values:** None
- **Train / Test split:** 80% / 20% (stratified, `random_state=42`)
- **Preprocessing:** `StandardScaler` fitted on training data only

---

## c. GitHub Repository Link
🔗 **[https://github.com/2026aj05211/phishing-website-classification-2026AJ05211](https://github.com/2026aj05211/phishing-website-classification-2026AJ05211)**

---

## d. Models Used

All five classifiers were trained on the same dataset and evaluated on the same 20% hold-out test set.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9285 | 0.9808 | 0.9234 | 0.9504 | 0.9367 | 0.8551 |
| Decision Tree | 0.9711 | 0.9805 | 0.9702 | 0.9781 | 0.9741 | 0.9413 |
| KNN | 0.9480 | 0.9867 | 0.9486 | 0.9586 | 0.9535 | 0.8945 |
| Naive Bayes | 0.5975 | 0.9702 | 1.0000 | 0.2770 | 0.4338 | 0.3810 |
| Support Vector Machine | 0.9516 | 0.9893 | 0.9418 | 0.9732 | 0.9573 | 0.9022 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong baseline with high accuracy (92.85%) and AUC (0.9808). It performs well because the dataset allows for a roughly linear decision boundary once the features are standardized. |
| Decision Tree | **Best overall accuracy (97.11%) and MCC (0.9413).** It perfectly captures the complex, rule-based nature of URL features without needing feature scaling. |
| KNN | Achieved high accuracy (94.80%) and AUC (0.9867). The standardization step was crucial here, as KNN is highly sensitive to feature scaling. |
| Naive Bayes | Performed poorly on Accuracy (59.75%) and Recall (27.70%), though it achieved 100% Precision. This highlights the violation of the feature-independence assumption; it only predicted "Legitimate" when highly confident, missing most phishing sites. |
| Support Vector Machine | Achieved the **highest AUC score (0.9893)** and strong accuracy (95.16%). The RBF kernel successfully handled the non-linear boundaries, making it highly reliable at distinguishing classes. |
| **Overall Winner** | **Decision Tree** — Achieved the highest Accuracy (97.11%) and MCC (0.9413). However, SVM had the best overall AUC and generalization balance. |

---

## e. Streamlit Application
🔗 **Live App:** [https://phishing-classifier-2026aj05211.streamlit.app](https://phishing-classifier-2026aj05211.streamlit.app)

**Features implemented:**
- 📂 CSV upload of test data (30 features + `Result` column)
- 🔽 Model-selection dropdown (all 5 classifiers)
- 📊 Display of all 6 evaluation metrics — Accuracy, AUC, Precision, Recall, F1, MCC
- 🔢 Confusion matrix (heatmap) and full classification report
- ✅ Models loaded from disk — **no retraining** on uploaded data

---

## Repository Structure
```text
.
├── app.py                          # Streamlit application
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
├── test_data.csv                   # 20% hold-out set with labels
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── scaler.joblib               # StandardScaler used in training
