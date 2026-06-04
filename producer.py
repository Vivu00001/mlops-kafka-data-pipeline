from kafka import KafkaProducer
import json
import random
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

merchants = [
    "Amazon",
    "Flipkart",
    "Myntra",
    "Swiggy",
    "Zomato"
]

print("Starting to send transactions...\n")

for i in range(100):

    transaction = {
        "user_id": random.randint(1, 100),
        "amount": round(random.uniform(100, 10000), 2),
        "merchant": random.choice(merchants),
        "timestamp": int(time.time())    # ← added
    }

    producer.send("transactions", transaction)

    print(f"Transaction {i+1}/100 → user_id={transaction['user_id']}  merchant={transaction['merchant']}  amount=₹{transaction['amount']}  timestamp={transaction['timestamp']}")

    time.sleep(0.1)   # ← changed from 1s to 0.1s (finishes in 10s instead of 100s)

producer.flush()

print("\n✓ Successfully sent 100 transactions to Kafka topic 'transactions'.")