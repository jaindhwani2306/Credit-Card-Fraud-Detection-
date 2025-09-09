# Credit-Card-Fraud-Detection-
This project focuses on building a machine learning model to detect fraudulent credit card transactions. The goal is to identify suspicious activity with high accuracy, minimizing both false positives (legitimate transactions flagged as fraud) and false negatives (fraudulent transactions missed by the model)
The solution involves a complete end-to-end pipeline, from data exploration and cleaning to training and evaluating two powerful machine learning classifiers: Random Forest and XGBoost.

2. Table of Contents
If your README is long, a table of contents with links to different sections helps with navigation.

3. Data Sources
 Dataset link : - ["https://www.kaggle.com/datasets/kartik2112/fraud-detection/data"]
The dataset used in this project is from Kaggle and contains a large number of credit card transactions.

The dataset includes both legitimate and fraudulent transactions, along with various features such as transaction amount, merchant details, and cardholder information.

4. Setup and Installation
Install the required libraries:
 pandas
 scikit-learn
 xgboost
 matplotlib
 seaborn

5. SQL Queries
Before the machine learning pipeline, several SQL queries were used for initial data exploration and preprocessing. These queries were instrumental in cleaning the raw data, handling missing values, and preparing the dataset for further analysis.
example:
-- Query to count the number of fraudulent and legitimate transactions
SELECT is_fraud, COUNT(*)
FROM transactions_table
GROUP BY is_fraud;

-- Query to calculate the average transaction amount for each category
SELECT category, AVG(amt)
FROM transactions_table
GROUP BY category
ORDER BY AVG(amt) DESC;

2. Fraud Rates by Transaction Category
This query identifies which transaction categories have the highest fraud rates, providing insight into which types of purchases are most targeted by fraudsters.

SQL

SELECT
  category,
  COUNT(*) AS total_transactions,
  SUM(is_fraud) AS fraud_transactions,
  ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_percent
FROM fraudTrain
GROUP BY category
ORDER BY fraud_rate_percent DESC;

3. Top Merchants with Most Fraudulent Transactions
This query helps spot specific merchants with a high count of fraudulent transactions.

SQL

SELECT
  merchant,
  COUNT(*) AS total_transactions,
  SUM(is_fraud) AS fraud_transactions,
  ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_percent
FROM fraudTrain
GROUP BY merchant
ORDER BY fraud_transactions DESC
LIMIT 10;

4. Cities with High Fraud Rates
This query identifies cities with a suspiciously high fraud rate, after filtering for cities with a minimum number of transactions to ensure the results are statistically significant.

SQL

SELECT
  city,
  COUNT(*) AS total_transactions,
  SUM(is_fraud) AS fraud_transactions,
  ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate_percent
FROM fraudTrain
GROUP BY city
HAVING total_transactions > 1000
ORDER BY fraud_rate_percent DESC
LIMIT 10;

5. Daily Transaction and Fraud Trend
This query shows how the total number of transactions and the fraud rate change over time.

SQL

SELECT DATE(trans_date_trans_time) AS txn_date,
       COUNT(*) AS total_txns,
       SUM(is_fraud) AS frauds,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS fraud_rate
FROM fraudTrain
GROUP BY txn_date
ORDER BY txn_date;

7. Code Structure and Methodology
The project follows a standard machine learning workflow:

Data Preprocessing: The raw data was cleaned, and categorical features (category, gender, city, state) were converted into a numerical format using one-hot encoding.

Feature Engineering: New features, such as hour, day_of_week, and customer_age, were engineered from existing date and time columns to provide more context for the models.

Model Training: Two models, Random Forest and XGBoost, were trained on the preprocessed data. Both models are robust for imbalanced classification tasks like fraud detection.

Model Evaluation: The models' performance was evaluated using a classification report, which provides key metrics like precision, recall, and F1-score. A feature importance plot was also generated to understand which features were most influential in the Random Forest model's predictions.

7. Results and Visualizations
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


