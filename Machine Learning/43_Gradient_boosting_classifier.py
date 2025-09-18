import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 45, 52, 23, 40, 60, 30, 35, 48, 50],
    'Salary': [40000, 80000, 75000, 30000, 60000, 90000, 50000, 55000, 70000, 65000],
    'Tenure': [1, 10, 5, 2, 8, 12, 3, 4, 7, 9],
    'Churn': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)

X = df[['Age', 'Salary', 'Tenure']]
y = df['Churn']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Raport:", classification_report(y_test, y_pred))