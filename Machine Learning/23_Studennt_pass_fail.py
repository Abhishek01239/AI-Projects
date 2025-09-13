import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
})

X = data[["Hours_Studied"]]
y = data["Pass"]

X_train,  X_test,y_train, y_test  = train_test_split(X, y, test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict([[12]])
print("Model Prediction for 12 hour study:" , y_pred)