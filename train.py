import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ================================================
# STEP 1 — Load processed features
# ================================================
print("=" * 50)
print("Loading data...")
print("=" * 50)

df = pd.read_csv("data/processed/features.csv")
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ================================================
# STEP 2 — Define features and target
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

print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")
print(f"Class distribution:\n{y.value_counts()}")

# ================================================
# STEP 3 — Train test split
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"\nTrain size: {len(X_train)} rows")
print(f"Test size:  {len(X_test)} rows")

# ================================================
# STEP 4 — Connect to MLflow server
# ================================================
# If MLflow running in Docker → use http://localhost:5000
# If MLflow running locally  → use http://localhost:5000
# If no MLflow server        → remove this line (saves locally)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("transaction_classification")
print("\nMLflow tracking URI: http://localhost:5000")
print("Experiment: transaction_classification")

# ================================================
# STEP 5 — Define 3 models to compare
# ================================================
models = {
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=200,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ),
        "params": {
            "max_iter": 200,
            "C": 1.0,
            "solver": "lbfgs"
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        "params": {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=42
        ),
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2
        }
    }
}

# ================================================
# STEP 6 — Train, evaluate, and log each model
# ================================================
print("\n" + "=" * 50)
print("Training and logging 3 models...")
print("=" * 50)

results = {}
best_model      = None
best_f1         = -1
best_model_name = ""

for model_name, model_info in models.items():
    print(f"\n--- {model_name} ---")

    model  = model_info["model"]
    params = model_info["params"]

    with mlflow.start_run(run_name=model_name):

        # --- Train ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Calculate all metrics ---
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)

        tn = int(cm[0][0]) if cm.shape[0] > 1 else 0
        fp = int(cm[0][1]) if cm.shape[1] > 1 else 0
        fn = int(cm[1][0]) if cm.shape[0] > 1 else 0
        tp = int(cm[1][1]) if cm.shape == (2,2) else 0

        # --- Log parameters ---
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))
        mlflow.log_param("features",     str(feature_cols))

        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # --- Log metrics ---
        mlflow.log_metric("accuracy",   accuracy)
        mlflow.log_metric("precision",  precision)
        mlflow.log_metric("recall",     recall)
        mlflow.log_metric("f1_score",   f1)
        mlflow.log_metric("true_positive",  tp)
        mlflow.log_metric("true_negative",  tn)
        mlflow.log_metric("false_positive", fp)
        mlflow.log_metric("false_negative", fn)

        # --- Log model artifact ---
        mlflow.sklearn.log_model(model, model_name)

        # --- Print results ---
        print(f"  Accuracy:        {accuracy:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  F1 Score:        {f1:.4f}")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")

        # --- Store results for summary ---
        results[model_name] = {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1
        }

        # --- Track best model by F1 score ---
        if f1 > best_f1:
            best_f1         = f1
            best_model      = model
            best_model_name = model_name

# ================================================
# STEP 7 — Print comparison summary
# ================================================
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)
for name, metrics in results.items():
    marker = " ← BEST" if name == best_model_name else ""
    print(f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f}{marker}")

# ================================================
# STEP 8 — Save best model
# ================================================
joblib.dump(best_model, "model.pkl")

print(f"\nBest model : {best_model_name}")
print(f"Best F1    : {best_f1:.4f}")
print(f"Saved to   : model.pkl")
print("\n" + "=" * 50)
print("Done. Task 7 complete.")
print("=" * 50)
print("\nTo view MLflow dashboard:")
print("  mlflow ui  OR  python -m mlflow ui")
print("  Open: http://127.0.0.1:5000")