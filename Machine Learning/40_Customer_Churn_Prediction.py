import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    "Tenure": [1, 12, 24, 36, 48, 60],
    "MonthlyCharges": [70, 80, 90, 100, 110, 120],
    "Churn": [1, 0, 0, 0, 1, 1]  # 1 = Churned, 0 = Stayed
})

X = data[["Tenure", "MonthlyCharges"]]
y = data["Churn"]

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Prediction (Tenure = 43, MonthlyCharges=120):", model.predict([[43, 120]]))