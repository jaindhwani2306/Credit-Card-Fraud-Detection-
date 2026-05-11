# Credit-Card-Fraud-Detection-
This project focuses on building a machine learning model to detect fraudulent credit card transactions. The goal is to identify suspicious activity with high accuracy, minimizing both false positives (legitimate transactions flagged as fraud) and false negatives (fraudulent transactions missed by the model)
The solution involves a complete end-to-end pipeline, from data exploration and cleaning to training and evaluating two powerful machine learning classifiers: Random Forest and XGBoost.


##Results and Visualizations
The classification_report for both models and the feature importance plot are shown below.

Random Forest Classifier Results:
[ Random Forest Results:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      9960
           1       1.00      0.50      0.67        40

    accuracy                           1.00     10000
   macro avg       1.00      0.75      0.83     10000
weighted avg       1.00      1.00      1.00     10000]

XGBoost Classifier Results:
[precision    recall  f1-score   support

           0       1.00      1.00      1.00      9960
           1       0.94      0.85      0.89        40

    accuracy                           1.00     10000
   macro avg       0.97      0.92      0.95     10000
weighted avg       1.00      1.00      1.00     10000]
Top 10 Feature Importances


