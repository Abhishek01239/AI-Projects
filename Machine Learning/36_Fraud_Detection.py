import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "Amount": [100, 2000, 5000, 150, 12000, 50],
    "Time": [1, 50, 200, 5, 300, 2],
    "Fraud": [0,1,1,0,1,0]
})

X = data[["Amount", "Time"]]
y = data["Fraud"]

X_train, X_test, y_train,y_test  = train_test_split(X,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Prediction (Amount = 2500 , Time = 40)", model.predict([[2500, 40]]))