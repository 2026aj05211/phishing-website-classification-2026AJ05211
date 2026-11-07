# Machine Learning Assignment 2

**Name:** Atanu Biswas  
**BITS ID:** 2026AJ05211  

### a. Problem statement
Phishing attacks are a major cybersecurity threat where attackers create fraudulent websites to steal sensitive information such as login credentials, credit-card numbers, and personal data. This project builds and compares five machine learning classification models that distinguish legitimate websites from phishing websites using 30 URL and webpage features extracted from the UCI Phishing Websites dataset. The trained models are deployed in an interactive Streamlit application that evaluates uploaded test data without retraining.

### b. Dataset description
- **Source:** UCI Machine Learning Repository — Phishing Websites Dataset (ID 327)
- **Instances:** 2,456
- **Features:** 30 (all numeric, integer-encoded: −1 / 0 / 1)
- **Target:** Binary — `Result` (1 = legitimate, 0 = phishing)
- **Subject Area:** Computer Science / Cybersecurity
- **Missing values:** None
- **Train / Test split:** 80% / 20% (stratified, `random_state=42`)
- **Preprocessing:** `StandardScaler` fitted on training data only

### c. Github Repository Link
🔗 **https://github.com/2026aj05211/phishing-website-classification-2026AJ05211**

### d. Models used

**Comparison Table:**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9285 | 0.9808 | 0.9234 | 0.9504 | 0.9367 | 0.8551 |
| Decision Tree | 0.9711 | 0.9805 | 0.9702 | 0.9781 | 0.9741 | 0.9413 |
| KNN | 0.9480 | 0.9867 | 0.9486 | 0.9586 | 0.9535 | 0.8945 |
| Naive Bayes | 0.5975 | 0.9702 | 1.0000 | 0.2770 | 0.4338 | 0.3810 |
| Support Vector Machine | 0.9516 | 0.9893 | 0.9418 | 0.9732 | 0.9573 | 0.9022 |

**Observations on the performance of each model:**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong baseline with high accuracy (92.85%) and AUC (0.9808). It performs well because the dataset allows for a roughly linear decision boundary once the features are standardized. |
| Decision Tree | Best overall accuracy (97.11%) and MCC (0.9413). It perfectly captures the complex, rule-based nature of URL features without needing feature scaling. |
| KNN | Achieved high accuracy (94.80%) and AUC (0.9867). The standardization step was crucial here, as KNN is highly sensitive to feature scaling. |
| Naive Bayes | Performed poorly on Accuracy (59.75%) and Recall (27.70%), though it achieved 100% Precision. This highlights the violation of the feature-independence assumption; it only predicted "Legitimate" when highly confident, missing most phishing sites. |
| Support Vector Machine | Achieved the highest AUC score (0.9893) and strong accuracy (95.16%). The RBF kernel successfully handled the non-linear boundaries, making it highly reliable at distinguishing classes. |
| Overall, Winner for your dataset | **Decision Tree** — Achieved the highest Accuracy (97.11%) and MCC (0.9413). However, SVM had the best overall AUC and generalization balance. |

### e. Streamlit Application
🔗 **Live App:** [https://phishing-classifier-2026aj05211.streamlit.app/](https://phishing-classifier-2026aj05211.streamlit.app/)

**Features implemented:**
- 📂 CSV upload of test data (30 features + `Result` column)
- 🔽 Model-selection dropdown (all 5 classifiers)
- 📊 Display of all 6 evaluation metrics — Accuracy, AUC, Precision, Recall, F1, MCC
- 🔢 Confusion matrix (heatmap) and full classification report
- ✅ Models loaded from disk — **no retraining** on uploaded data

---

### f. Model Performance Justification

Below is a detailed justification of why each model performed the way it did on the UCI Phishing Websites dataset:

1. **Logistic Regression:**
   - **Why it performed well:** Logistic Regression achieved a high accuracy of 92.85% because the 30 features in this dataset are designed to be highly discriminative. Once the features are standardized using `StandardScaler`, the classes are largely linearly separable. Furthermore, Logistic Regression outputs well-calibrated probabilities, which is why its AUC is exceptionally high (0.9808).
   - **Limitations:** It cannot naturally capture complex, non-linear interactions between features (e.g., how a specific combination of an abnormal URL and a missing SSL state drastically increases phishing risk) without manual feature engineering.

2. **Decision Tree:**
   - **Why it performed the best:** The Decision Tree achieved the highest Accuracy (97.11%) and MCC (0.9413). This is because the features in the dataset (like `SSLfinal_State` or `URL_of_Anchor`) are categorical (represented as -1, 0, 1). Decision Trees excel at creating rule-based splits for exactly this type of data. It does not require feature scaling to perform optimally.
   - **Limitations:** Decision Trees are prone to overfitting. However, because we used a stratified 80/20 split and the dataset is relatively small (2,456 rows), it generalized well.

3. **K-Nearest Neighbors (KNN):**
   - **Why it performed well:** KNN achieved 94.80% accuracy and a very high AUC (0.9867). The key to KNN's success here was the `StandardScaler`. Because KNN relies on Euclidean distance, unscaled features would have dominated the distance calculations. Once standardized, the nearest neighbors accurately reflected the true class of the test instances.
   - **Limitations:** KNN is computationally expensive during prediction because it must calculate the distance to all training samples. It is also highly sensitive to irrelevant features (the "curse of dimensionality").

4. **Naive Bayes (Gaussian):**
   - **Why it performed poorly:** Naive Bayes had the lowest accuracy (59.75%) and recall (27.70%). The core assumption of Naive Bayes is that all features are **conditionally independent** given the class. In cybersecurity, this is a severe violation. For example, the presence of an `Abnormal_URL` is highly correlated with the `SSLfinal_State`. Because it assumes independence, the model became overly conservative—it achieved 100% precision (it never falsely accused a legitimate site) but completely failed to identify most phishing sites (very low recall).

5. **Support Vector Machine (SVM):**
   - **Why it performed exceptionally well:** SVM achieved the highest AUC (0.9893) and a very strong accuracy of 95.16%. By using the **RBF (Radial Basis Function) kernel**, the SVM was able to project the data into a higher-dimensional space where the non-linear relationships between the URL features could be perfectly separated by a hyperplane. It provides the best balance of high precision and high recall, making it a highly robust model for this problem.

**Overall Conclusion:** While the Decision Tree is the winner based on raw Accuracy and MCC, the **SVM is arguably the most robust model** for this specific problem due to its superior AUC score and balanced precision/recall, which are critical in cybersecurity where false positives and false negatives both carry heavy costs.
