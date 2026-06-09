import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

df = pd.read_csv("data/raw/realistic_data.csv")
print(f"Loaded {len(df)} rows")
print(f"Class distribution:\n{df['is_weekend'].value_counts()}")

# Encode merchant
le = LabelEncoder()
df['merchant_encoded'] = le.fit_transform(df['merchant'])
print(f"Merchant encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Scale amount and user_id
scaler_amount  = StandardScaler()
scaler_user    = StandardScaler()

df['amount_scaled']  = scaler_amount.fit_transform(df[['amount']])
df['user_id_scaled'] = scaler_user.fit_transform(df[['user_id']])

# Time features
df['is_night']    = df['hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)
df['is_lunch']    = df['hour'].apply(lambda x: 1 if 12 <= x <= 14 else 0)
df['is_evening']  = df['hour'].apply(lambda x: 1 if 17 <= x <= 20 else 0)

# Drop columns not needed
df.drop(columns=['merchant', 'timestamp'], inplace=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/realistic_features.csv", index=False)

print(f"\nFinal columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print(df.head())
print("\nDone.")
