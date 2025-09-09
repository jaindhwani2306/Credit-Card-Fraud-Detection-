import pandas as pd
from sqlalchemy import create_engine

# 1. Load the CSV file
df = pd.read_csv('C:/Users\Dell/Documents/my projects/credit card fraud detection/fraudTest.csv')

# 2. Basic cleaning
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]  # clean column names

# Convert datetime string to datetime object
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Optional: Drop duplicates or nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 3. Create SQLite DB and connect
engine = create_engine('sqlite:///fraud_detection.db')  # Creates fraud_detection.db in current dir

# 4. Load DataFrame into SQLite
df.to_sql('fraud_train', con=engine, if_exists='replace', index=False)

print(f"✅ Loaded {len(df)} rows into the 'fraud_train' table in fraud_detection.db")
