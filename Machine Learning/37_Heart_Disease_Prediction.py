import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier

data = pd.DataFrame({
    "Age": [29, 45, 50, 60, 35, 40],
    "Cholesterol": [200, 240, 300, 280, 180, 220],
    "Heart_Disease": [0,1,1,1,0,0]
})


X = data[["Age", "Cholesterol"]]
y = data["Heart_Disease"]

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

print("Prediction (Age = 55, Cholesterol = 250):", model.predict([[155, 250]]))
