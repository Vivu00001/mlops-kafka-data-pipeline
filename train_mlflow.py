import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================================
# STEP 1 - Load features
# ================================================
print("="*60)
print("STEP 1 - Loading data")
print("="*60)
df = pd.read_csv("data/processed/features.csv")
print(f"Loaded {len(df)} rows")

# ================================================
# STEP 2 - Fix class imbalance
# ================================================
print("\nSTEP 2 - Fixing class imbalance")
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

if df['is_weekend'].sum() == 0:
    print("Simulating weekend rows...")
    weekend_rows = df.sample(n=30, random_state=42).copy()
    weekend_rows['day_of_week'] = np.random.choice([5, 6], size=30)
    weekend_rows['is_weekend'] = 1
    df = pd.concat([df, weekend_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Class distribution:\n{df['is_weekend'].value_counts()}")

# ================================================
# STEP 3 - Features and target
# ================================================
feature_cols = [
    'user_id_scaled',
    'amount_scaled',
    'merchant_encoded',
    'hour',
    'day_of_week'
]
target_col = 'is_weekend'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nTrain: {len(X_train)} rows, Test: {len(X_test)} rows")

# ================================================
# STEP 4 - MLflow setup
# ================================================
print("\nSTEP 4 - Setting up MLflow")
TRACKING_URI  = "sqlite:///mlflow.db"
EXPERIMENT    = "transaction_classification"
REGISTRY_NAME = "TransactionModel"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient(tracking_uri=TRACKING_URI)
print(f"Tracking URI : {TRACKING_URI}")
print(f"Experiment   : {EXPERIMENT}")
print(f"Registry name: {REGISTRY_NAME}")

# ================================================
# STEP 5 - Define 3 models
# ================================================
models = {
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=200, C=1.0, random_state=42
        ),
        "params": {"max_iter": 200, "C": 1.0}
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "params": {"max_depth": 5}
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "params": {"n_estimators": 100, "max_depth": 5}
    }
}

# ================================================
# STEP 6 - Train, evaluate, log, register
# ================================================
print("\n" + "="*60)
print("STEP 6 - Training and registering 3 models")
print("="*60)

results         = {}
best_model      = None
best_f1         = -1
best_model_name = ""
best_run_id     = ""

for model_name, model_info in models.items():
    print(f"\n--- Training {model_name} ---")

    model  = model_info["model"]
    params = model_info["params"]

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        # Log params
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        # Log and register model — CORRECTLY INDENTED inside with block
        mlflow.sklearn.log_model(
            sk_model              = model,
            name                  = model_name,
            registered_model_name = REGISTRY_NAME
        )

        # Print results
        print(f"  Run ID:    {run_id}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Registered as: {REGISTRY_NAME}")

        results[model_name] = {
            "run_id":    run_id,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1
        }

        if f1 > best_f1:
            best_f1         = f1
            best_model      = model
            best_model_name = model_name
            best_run_id     = run_id

# ================================================
# STEP 7 - If all models tie prefer RandomForest
# ================================================
lr_f1 = results.get("LogisticRegression", {}).get("f1_score", 0)
dt_f1 = results.get("DecisionTree",       {}).get("f1_score", 0)
rf_f1 = results.get("RandomForest",       {}).get("f1_score", 0)

if lr_f1 == dt_f1 == rf_f1:
    best_model_name = "RandomForest"
    best_run_id     = results["RandomForest"]["run_id"]
    best_model      = models["RandomForest"]["model"]
    print("\nAll models tied - defaulting to RandomForest as best model")

# ================================================
# STEP 8 - Promote best model to Production
# ================================================
print("\n" + "="*60)
print("STEP 8 - Promoting best model to Production")
print("="*60)

all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")

print(f"\nAll registered versions:")
for v in all_versions:
    print(f"  Version {v.version} | Run: {v.run_id[:8]}... | Stage: {v.current_stage}")

best_version = None
for v in all_versions:
    if v.run_id == best_run_id:
        best_version = v.version
        break

if best_version:
    # Archive existing production
    for v in all_versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name    = REGISTRY_NAME,
                version = v.version,
                stage   = "Archived"
            )
            print(f"  Archived version {v.version}")

    # Promote to production
    client.transition_model_version_stage(
        name    = REGISTRY_NAME,
        version = best_version,
        stage   = "Production"
    )
    print(f"\nPromoted version {best_version} ({best_model_name}) to Production")

    client.update_model_version(
        name        = REGISTRY_NAME,
        version     = best_version,
        description = f"Best model: {best_model_name} | F1={best_f1:.4f}"
    )

# ================================================
# STEP 9 - Load Production model from Registry
# ================================================
print("\n" + "="*60)
print("STEP 9 - Loading Production model from Registry")
print("="*60)

production_uri = f"models:/{REGISTRY_NAME}/Production"
loaded_model   = mlflow.sklearn.load_model(production_uri)
print(f"Loaded from: {production_uri}")

sample = X_test.iloc[:3]
preds  = loaded_model.predict(sample)
print(f"Sample predictions: {preds}")

# ================================================
# STEP 10 - Comparison summary
# ================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*60)
for name, m in results.items():
    marker = " << PRODUCTION" if name == best_model_name else ""
    print(f"{name:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}{marker}")

# ================================================
# STEP 11 - Save best model
# ================================================
joblib.dump(best_model, "model.pkl")
print(f"\nBest model : {best_model_name}")
print(f"Best F1    : {best_f1:.4f}")
print(f"Saved      : model.pkl")
print(f"Registry   : {REGISTRY_NAME} -> Production")
print("="*60)
print("\nOpen MLflow UI:")
print("  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("  http://127.0.0.1:5000")
print("  Go to Models tab to see Registry")