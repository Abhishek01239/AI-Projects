import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    "Temp": [30, 25, 20, 15, 10, 5],
    "Humidity": [80, 70, 65, 60, 55, 50],
    "Rain": [1, 1, 0, 0, 0, 0]
})

X = data[["Temp", "Humidity"]]
y = data["Rain"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Prediction (Temp=22, Humidity=68):", model.predict([[22, 68]]))
