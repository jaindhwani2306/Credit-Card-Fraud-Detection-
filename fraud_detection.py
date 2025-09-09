import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set the file path
file_path = "C:/Users/Dell/Documents/my projects/credit card fraud detection/fraudTest.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f" Error: The file '{file_path}' was not found.")
    exit()


# Data Loading and Preprocessing

try:
    # Load a subset of the data for faster testing
    df = pd.read_csv(file_path, nrows=50000)
    print("Data loaded successfully")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# Lowercase columns and preprocess datetime
df.columns = df.columns.str.lower().str.strip()
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Feature engineering
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
# Use errors='coerce' to handle potential non-numeric values
df['customer_age'] = 2020 - pd.to_numeric(df['dob'].str[:4], errors='coerce')
# Drop unnecessary columns
columns_to_drop = [
    'trans_date_trans_time', 'merchant', 'first', 'last', 'street',
    'job', 'dob', 'unix_time', 'cc_num', 'trans_num'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['category', 'gender', 'city', 'state'], drop_first=True)



X = df.drop('is_fraud', axis=1)
y = df['is_fraud']


non_numeric_cols = X.select_dtypes(include=['object']).columns
if not non_numeric_cols.empty:
    print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
    X = X.drop(columns=non_numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Random Forest Model with parallel processing
print("\nTraining Random Forest Model")

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# XGBoost Model with parallel processing
print("\nTraining XGBoost Model")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb, zero_division=0))


# Visualization

try:
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    print("\nFeature importance plot displayed.")
except Exception as e:
    print(f"\nAn error occurred while generating the feature importance plot: {e}")

print("\nScript completed.")