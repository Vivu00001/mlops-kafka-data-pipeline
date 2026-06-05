import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# --- Load validated data ---
df = pd.read_csv("data/validated/data.csv")
print(f"Loaded {len(df)} rows")
print(df.head())

# --- Feature 1: Encode merchant (text → number) ---
le = LabelEncoder()
df['merchant_encoded'] = le.fit_transform(df['merchant'])
print(f"\nMerchant encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Feature 2: Extract time features from timestamp ---
df['datetime']    = pd.to_datetime(df['timestamp'], unit='s')
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend']  = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# --- Feature 3: Scale amount and user_id ---
scaler = StandardScaler()
df['amount_scaled']  = scaler.fit_transform(df[['amount']])
df['user_id_scaled'] = scaler.fit_transform(df[['user_id']])

# --- Drop columns not needed for ML ---
df.drop(columns=['datetime', 'merchant', 'timestamp'], inplace=True)

# --- Save processed features ---
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/features.csv", index=False)

print(f"\nFeature columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print(f"\nSample:")
print(df.head())
print("\nDone. Task 5 complete.")