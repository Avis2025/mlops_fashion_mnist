import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST and Increase Dataset Size
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test, y_train, y_test = X_train[:1000], X_test[:300], y_train[:1000], y_test[:300]  # Increased size

# Flatten and Normalize
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Split into Train/Validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define Model Selection Using XGBoost
model = xgb.XGBClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
model.fit(X_train, y_train)
print("Baseline Model Accuracy:", accuracy_score(y_val, model.predict(X_val)))

# Hyperparameter Tuning using Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 150, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 12, step=3)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.05)
    
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

# Train Best Model with Optimized Hyperparameters
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=1)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Final Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy:.4f}")
print("Best Hyperparameters:", best_params)

# Explainability with SHAP
explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_test[:50])  # Explain a small batch
shap.summary_plot(shap_values, X_test[:50])
