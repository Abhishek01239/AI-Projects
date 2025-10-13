import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': model.feature_importances_
}).sort_values(by = "Importance", ascending = False)

plt.figure(figsize=(10,6))
sns.barplot(x = "Importance", y = "Feature", data = feature_importance, palette='coolwarm')
plt.title("Random Forest Feature Importance")
plt.show()

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

perm_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values(by = "Importance", ascending=False)

plt.figure(figsize = (10, 6))
sns.barplot(x = "Importance", y = "Feature", data = perm_df, palette="viridis")
plt.title("Permutation Feature Importance")
plt.show()