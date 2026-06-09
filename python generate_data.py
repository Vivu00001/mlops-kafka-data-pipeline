import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

np.random.seed(42)
random.seed(42)

print("Generating realistic transaction data...")

# ================================================
# Realistic patterns in real e-commerce data:
# - Weekend transactions are higher amount
# - Swiggy/Zomato orders peak at night (hour 20-23)
# - Amazon/Flipkart peak on weekdays
# - User spending varies by merchant
# ================================================

merchants = ["Amazon", "Flipkart", "Myntra", "Swiggy", "Zomato"]
n_rows    = 1000   # much more data than 137

rows = []

# Generate data across 90 days (mix of weekdays and weekends)
start_date = datetime(2024, 1, 1)

for i in range(n_rows):
    # Random day within 90 days
    day_offset  = random.randint(0, 89)
    current_day = start_date + timedelta(days=day_offset)
    day_of_week = current_day.weekday()   # 0=Monday, 6=Sunday
    is_weekend  = 1 if day_of_week >= 5 else 0

    # Realistic hour patterns
    if is_weekend:
        # Weekends — people shop more in afternoon/evening
        hour = random.choices(
            range(24),
            weights=[1,1,1,1,1,1,2,3,4,5,6,7,8,9,10,10,9,8,9,10,9,8,5,3],
            k=1
        )[0]
    else:
        # Weekdays — lunch break and after work
        hour = random.choices(
            range(24),
            weights=[1,1,1,1,1,2,3,4,5,6,7,8,9,10,8,7,8,9,10,9,7,5,3,2],
            k=1
        )[0]

    # Merchant selection — realistic patterns
    if is_weekend:
        # Weekends more food delivery
        merchant = random.choices(
            merchants,
            weights=[20, 15, 15, 25, 25],
            k=1
        )[0]
    else:
        # Weekdays more shopping
        merchant = random.choices(
            merchants,
            weights=[30, 30, 20, 10, 10],
            k=1
        )[0]

    # Amount — realistic patterns
    base_amounts = {
        "Amazon":   (500,  8000),
        "Flipkart": (300,  6000),
        "Myntra":   (400,  5000),
        "Swiggy":   (150,  800),
        "Zomato":   (150,  900)
    }
    low, high = base_amounts[merchant]

    # Weekend premium — people spend more on weekends
    if is_weekend:
        amount = round(random.uniform(low * 1.2, high * 1.3), 2)
    else:
        amount = round(random.uniform(low, high), 2)

    # Add some noise
    amount = max(50, amount + random.gauss(0, 50))
    amount = round(amount, 2)

    user_id   = random.randint(1, 200)
    timestamp = int(current_day.replace(hour=hour).timestamp())

    rows.append({
        "user_id":    user_id,
        "merchant":   merchant,
        "amount":     amount,
        "timestamp":  timestamp,
        "hour":       hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend
    })

df = pd.DataFrame(rows)

print(f"Generated {len(df)} rows")
print(f"\nClass distribution:")
print(df['is_weekend'].value_counts())
print(f"\nWeekend %: {df['is_weekend'].mean()*100:.1f}%")
print(f"\nMerchant distribution:")
print(df['merchant'].value_counts())
print(f"\nAmount stats:")
print(df['amount'].describe())

# Save
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/realistic_data.csv", index=False)
print(f"\nSaved to data/raw/realistic_data.csv")
