import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "Age" : [22, 38, 26, 35, 28, 2],
    "Fare" : [7.25, 71.83, 7.92, 53.1, 8.05, 21.07],
    "Survived":[0,1,1,1,0,1]
})

X = data[["Age", "Fare"]]
y = data["Survived"]

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Prediction (Age=30, Fare=10):", model.predict([[30,10]]))