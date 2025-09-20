import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Age': [25, 40, 35, 50, 29, 60, 45, 33, 52, 48],
    'Income': [30000, 80000, 60000, 120000, 40000, 150000, 95000, 70000, 110000, 85000],
    'CreditScore': [650, 720, 680, 800, 630, 850, 710, 690, 780, 750],
    'LoanAmount': [100000, 200000, 150000, 250000, 120000, 300000, 220000, 180000, 240000, 200000],
    'Approved': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

X = df[["Age", "Income", "CreditScore", "LoanAmount"]]
y = df["Approved"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder = False, eval_matric = 'logloss').fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))