import pandas as pd
import pickle

# --- Load features ---
df = pd.read_csv("data/processed/features.csv")
print(f"Loaded {len(df)} rows")

feature_cols = ['user_id_scaled', 'amount_scaled', 'merchant_encoded', 'hour', 'day_of_week']
target_col = 'is_weekend'

X = df[feature_cols]
y = df[target_col]

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Save threshold directly (no class needed) ---
threshold = 4
correct = ((X_test['day_of_week'] >= threshold).astype(int).values == y_test.values).sum()
accuracy = correct / len(y_test)
print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_test)})")

# --- Save just the threshold value ---
with open("model.pkl", "wb") as f:
    pickle.dump({"threshold": threshold}, f)

print("Model saved as model.pkl")
print("Done. Task 7 complete.")