# Project 55 - Hyperparameter Optimization for XGBoost using Optuna

import optuna
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define objective function
def objective(trial):
    params = {
        "eval_metric": "mlogloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss"
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# 3. Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 4. Results
print("\nBest Parameters Found:\n", study.best_params)
print(f"\nBest Accuracy: {study.best_value * 100:.2f}%")
