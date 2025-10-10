import time 
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "XGBoost": XGBClassifier(use_label_encodet = False, eval_metric = "mlogloss", n_estimators = 200, learning_rate = 0.05),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05),
    "CatBoost": CatBoostClassifier(iterations = 200, learning_rate= 0.05, verbose =0)
}

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)    
    duration = round(end - start,2)

    results.append({"Model": name, "Accuracy": acc*100, "Train Time (s)": duration})

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["Accuracy"], color = ["#1f77b4", "#2ca02c", "#ff7f0e"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()