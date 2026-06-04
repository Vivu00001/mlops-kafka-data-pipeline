from kafka import KafkaConsumer
import json
import psycopg2

# --- Kafka connection ---
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='mlops-group',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# --- PostgreSQL connection ---
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='kafka_mlops',
    user='postgres',
    password='postgres'
)
cursor = conn.cursor()

# --- Create table ---
cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        merchant VARCHAR(100),
        amount FLOAT,
        timestamp BIGINT
    )
""")
conn.commit()

print("Table ready. Listening for messages...")

# --- Consume and insert ---
for message in consumer:
    data = message.value
    print(f"Received: {data}")

    # Skip old messages that don't have timestamp
    if "timestamp" not in data:
        print(f"Skipping old message (no timestamp): {data}")
        continue

    cursor.execute("""
        INSERT INTO transactions (user_id, merchant, amount, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (data["user_id"], data["merchant"], data["amount"], data["timestamp"]))

    conn.commit()
    print(f"Saved → user_id={data['user_id']}, merchant={data['merchant']}, amount={data['amount']}")