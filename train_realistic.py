import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ================================================
# STEP 1 - Load realistic features
# ================================================
print("="*60)
print("STEP 1 - Loading realistic data")
print("="*60)

df = pd.read_csv("data/processed/realistic_features.csv")
print(f"Loaded {len(df)} rows")
print(f"\nClass distribution:")
print(df['is_weekend'].value_counts())
print(f"\nWeekend %: {df['is_weekend'].mean()*100:.1f}%")

# ================================================
# STEP 2 - Features and target
# ================================================
feature_cols = [
    'user_id_scaled',
    'amount_scaled',
    'merchant_encoded',
    'hour',
    'day_of_week',
    'is_night',
    'is_lunch',
    'is_evening'
]
target_col = 'is_weekend'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # balanced split
)
print(f"\nTrain: {len(X_train)} rows")
print(f"Test:  {len(X_test)} rows")
print(f"Train weekend %: {y_train.mean()*100:.1f}%")
print(f"Test  weekend %: {y_test.mean()*100:.1f}%")

# ================================================
# STEP 3 - MLflow setup
# ================================================
TRACKING_URI  = "sqlite:///mlflow.db"
EXPERIMENT    = "realistic_transaction_classification"
REGISTRY_NAME = "RealisticTransactionModel"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient(tracking_uri=TRACKING_URI)
print(f"\nMLflow experiment: {EXPERIMENT}")

# ================================================
# STEP 4 - Define 3 models with different params
# ================================================
models = {
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=500,
            C=0.1,              # regularization — prevents overfitting
            solver='lbfgs',
            random_state=42
        ),
        "params": {
            "max_iter": 500,
            "C": 0.1,
            "solver": "lbfgs"
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(
            max_depth=4,        # shallow tree — prevents overfitting
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        "params": {
            "max_depth": 4,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 5
        }
    }
}

# ================================================
# STEP 5 - Train, evaluate, log, register
# ================================================
print("\n" + "="*60)
print("STEP 5 - Training 3 models with MLflow tracking")
print("="*60)

results         = {}
best_model      = None
best_f1         = -1
best_model_name = ""
best_run_id     = ""

for model_name, model_info in models.items():
    print(f"\n--- {model_name} ---")

    model  = model_info["model"]
    params = model_info["params"]

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Train
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

        # Log params
        mlflow.log_param("model_type",   model_name)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features",   len(feature_cols))
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy",        accuracy)
        mlflow.log_metric("precision",       precision)
        mlflow.log_metric("recall",          recall)
        mlflow.log_metric("f1_score",        f1)
        mlflow.log_metric("true_positive",   int(tp))
        mlflow.log_metric("true_negative",   int(tn))
        mlflow.log_metric("false_positive",  int(fp))
        mlflow.log_metric("false_negative",  int(fn))

        # Register model
        mlflow.sklearn.log_model(
            sk_model              = model,
            name                  = model_name,
            registered_model_name = REGISTRY_NAME
        )

        # Print full report
        print(f"  Accuracy:        {accuracy:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  F1 Score:        {f1:.4f}")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")

        results[model_name] = {
            "run_id": run_id, "accuracy": accuracy,
            "precision": precision, "recall": recall,
            "f1_score": f1, "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn)
        }

        if f1 > best_f1:
            best_f1         = f1
            best_model      = model
            best_model_name = model_name
            best_run_id     = run_id

# ================================================
# STEP 6 - Promote best model to Production
# ================================================
print("\n" + "="*60)
print("STEP 6 - Model Registry - Promoting to Production")
print("="*60)

all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")

# Archive existing production
for v in all_versions:
    if v.current_stage == "Production":
        client.transition_model_version_stage(
            name=REGISTRY_NAME, version=v.version, stage="Archived"
        )
        print(f"Archived version {v.version}")

# Find and promote best
for v in all_versions:
    if v.run_id == best_run_id:
        client.transition_model_version_stage(
            name=REGISTRY_NAME, version=v.version, stage="Production"
        )
        client.update_model_version(
            name=REGISTRY_NAME, version=v.version,
            description=f"Best: {best_model_name} | F1={best_f1:.4f}"
        )
        print(f"Promoted version {v.version} ({best_model_name}) to Production")
        break

# ================================================
# STEP 7 - Load from Registry and predict
# ================================================
print("\n" + "="*60)
print("STEP 7 - Loading Production model from Registry")
print("="*60)

prod_uri     = f"models:/{REGISTRY_NAME}/Production"
loaded_model = mlflow.sklearn.load_model(prod_uri)
print(f"Loaded from: {prod_uri}")

sample = X_test.iloc[:5]
preds  = loaded_model.predict(sample)
actual = y_test.iloc[:5].values
print(f"\nSample predictions vs actual:")
for i, (p, a) in enumerate(zip(preds, actual)):
    status = "CORRECT" if p == a else "WRONG"
    label_p = "weekend" if p == 1 else "weekday"
    label_a = "weekend" if a == 1 else "weekday"
    print(f"  Row {i+1}: Predicted={label_p:7s} Actual={label_a:7s} [{status}]")

# ================================================
# STEP 8 - Comparison summary
# ================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'TP':>5} {'FP':>5} {'FN':>5}")
print("-"*75)
for name, m in results.items():
    marker = " << BEST" if name == best_model_name else ""
    print(f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>9.4f} {m['recall']:>9.4f} {m['f1_score']:>9.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}{marker}")

# ================================================
# STEP 9 - Save best model
# ================================================
joblib.dump(best_model, "model.pkl")
print(f"\nBest model : {best_model_name}")
print(f"Best F1    : {best_f1:.4f}")
print(f"Saved      : model.pkl")
print(f"Registry   : {REGISTRY_NAME} -> Production")
print("="*60)
print("\nOpen MLflow UI:")
print("  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("  Open: http://127.0.0.1:5000")
