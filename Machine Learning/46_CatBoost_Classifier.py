import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 40, 30, 50, 28, 45, 35, 38, 48, 33],
    'Department': ['Sales', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Sales', 'Finance', 'IT', 'HR'],
    'YearsAtCompany': [1, 10, 5, 20, 3, 15, 6, 12, 18, 7],
    'JobSatisfaction': [3, 2, 4, 1, 3, 2, 4, 3, 2, 5],
    'Attrition': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)

X = df[["Age", "Department", "YearsAtCompany", "JobSatisfaction"]]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations = 200, learning_rate=0.1, depth = 4, verbose = 0)
model.fit(X_train, y_train, cat_features=["Department"])

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

