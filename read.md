# Kafka Data Ingestion Pipeline

## Project Overview

This project demonstrates a basic real-time data ingestion pipeline using Apache Kafka and PostgreSQL.

The pipeline generates transaction data, sends it to a Kafka topic, consumes the messages, and stores them in a PostgreSQL database.

This project is designed to help beginners understand:

* Apache Kafka fundamentals
* Producer and Consumer architecture
* Real-time data streaming
* PostgreSQL integration
* Data Engineering and MLOps basics

---

## Architecture

```text
Producer
   ↓
Kafka Topic (transactions)
   ↓
Consumer
   ↓
PostgreSQL Database
```

---

## Tech Stack

* Python 3.12
* Apache Kafka
* PostgreSQL
* Docker Desktop
* kafka-python
* psycopg2-binary

---

## Project Structure

```text
kafka_demo_mlops_project/
│
├── docker-compose.yml
├── producer.py
├── consumer.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

### 1. Producer

The producer generates transaction records and sends them to the Kafka topic named:

```text
transactions
```

Example message:

```json
{
  "user_id": 45,
  "amount": 2500.75,
  "merchant": "Amazon"
}
```

---

### 2. Kafka Topic

Kafka acts as a message broker.

The topic stores incoming transaction events before they are consumed.

Topic Name:

```text
transactions
```

---

### 3. Consumer

The consumer reads transaction messages from Kafka.

Responsibilities:

* Receive transaction events
* Parse JSON data
* Insert records into PostgreSQL

---

### 4. PostgreSQL

Stores transaction data permanently.

Table:

```sql
transactions
```

Columns:

```sql
transaction_id
user_id
amount
merchant
created_at
```

---

## Setup Instructions

### Clone Repository

```bash
git clone <repository-url>
cd kafka_demo_mlops_project
```

---

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Start Docker Containers

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

---

### Create Kafka Topic

```bash
docker exec -it kafka \
/opt/kafka/bin/kafka-topics.sh \
--create \
--topic transactions \
--bootstrap-server localhost:9092 \
--partitions 1 \
--replication-factor 1
```

---

### Run Consumer

```bash
python consumer.py
```

---

### Run Producer

```bash
python producer.py
```

---

## Verify Data

Connect to PostgreSQL:

```bash
docker exec -it postgres_db psql -U postgres -d kafka_mlops
```

Check records:

```sql
SELECT * FROM transactions;
```

Count records:

```sql
SELECT COUNT(*) FROM transactions;
```

---

## Learning Outcomes

By completing this project, you will understand:

* What Apache Kafka is
* How Producers send data
* How Consumers receive data
* Real-time event streaming concepts
* PostgreSQL integration
* Data ingestion pipelines
* Foundations of Data Engineering
* Foundations of MLOps

---

## Future Improvements

* Add MLflow tracking
* Train a fraud detection model
* Add Apache Airflow orchestration
* Add Dockerized Python services
* Add monitoring and logging
* Deploy on AWS/GCP/Azure
* Integrate with Spark Streaming

---

## Author

Vivek Parmar

B.Tech Computer Engineering (AI & ML)

Learning MLOps, Data Engineering, and Machine Learning Systems.
