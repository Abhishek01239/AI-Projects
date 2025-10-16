import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))

print("Calculation Permutation Feature Importance...")
result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by = 'Importance Mean', ascending=False)

print("\n Feature Importances: \n", importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance Mean'], xerr = importance_df['Importance Std'])
plt.gca().invert_yaxis()
plt.title("Permutation Feature Importance (Random Forest)")
plt.xlabel("Mean Decrease in accuracy")
plt.ylabel("Feature")
plt.tight_layout()
plt.savfig("permutation_feature_importance.png", bbox_inches = "tight")
print("Plot saved -> permutation_feature_importance.png")