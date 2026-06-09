import pandas as pd
import os

# --- Load raw data ---
df = pd.read_csv("data/raw/data.csv")
print(f"Loaded {len(df)} rows from data/raw/data.csv")
print(f"Columns: {list(df.columns)}")
print(f"\nOriginal shape: {df.shape}")

# --- Check 1: Missing values ---
print("\n--- Check 1: Missing Values ---")
missing = df.isnull().sum()
print(missing)
df.dropna(inplace=True)
print(f"After dropping nulls: {len(df)} rows")

# --- Check 2: Duplicate records ---
print("\n--- Check 2: Duplicates ---")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
df.drop_duplicates(inplace=True)
print(f"After dropping duplicates: {len(df)} rows")

# --- Check 3: Data types ---
print("\n--- Check 3: Data Types ---")
print(df.dtypes)
df['user_id']   = df['user_id'].astype(int)
df['amount']    = df['amount'].astype(float)
df['timestamp'] = df['timestamp'].astype(int)
df['merchant']  = df['merchant'].astype(str)
print("Data types corrected.")

# --- Check 4: Invalid values ---
print("\n--- Check 4: Invalid Values ---")
invalid_amount   = df[df['amount'] <= 0]
invalid_user     = df[df['user_id'] <= 0]
invalid_merchant = df[df['merchant'].str.strip() == '']

print(f"Invalid amounts (<=0): {len(invalid_amount)}")
print(f"Invalid user_ids (<=0): {len(invalid_user)}")
print(f"Invalid merchants (empty): {len(invalid_merchant)}")

df = df[df['amount'] > 0]
df = df[df['user_id'] > 0]
df = df[df['merchant'].str.strip() != '']

print(f"After removing invalid values: {len(df)} rows")

# --- Save validated data ---
os.makedirs("data/validated", exist_ok=True)
df.to_csv("data/validated/data.csv", index=False)

print(f"\nSaved to data/validated/data.csv")
print(f"Final shape: {df.shape}")
print(f"\nSample:")
print(df.head())
print("\nDone. Task 4 complete.")