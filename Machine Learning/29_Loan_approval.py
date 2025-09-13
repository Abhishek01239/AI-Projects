import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.DataFrame({
    "Income":[30000, 50000, 70000,90000, 120000],
    "Credit_Score":[500, 650, 700, 750, 800],
    "Approved": [0,0,1,1,1]
})

X = data[["Income", "Credit_Score"]]
y = data["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

print("Prediction (Income= 80000,, Score=720):", model.predict([[80000,720]]))