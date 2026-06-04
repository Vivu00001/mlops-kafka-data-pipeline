import psycopg2
import pandas as pd
import os

# --- Connect to PostgreSQL ---
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='kafka_mlops',
    user='postgres',
    password='postgres'
)

print("Connected to PostgreSQL...")

# --- Read data ---
query = "SELECT * FROM transactions ORDER BY id"
df = pd.read_sql(query, conn)

print(f"Extracted {len(df)} rows from transactions table")
print(df.head())

# --- Create folder if not exists ---
os.makedirs("data/raw", exist_ok=True)

# --- Save to CSV ---
df.to_csv("data/raw/data.csv", index=False)

print(f"\nSaved to data/raw/data.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# --- Close connection ---
conn.close()
print("\nDone. Task 2 complete.")