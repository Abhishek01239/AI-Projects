import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Fake stock dataset
data = pd.DataFrame({
    "Days": [1, 2, 3, 4, 5, 6, 7],
    "Price": [100, 102, 104, 107, 110, 113, 115]
})

X = data[["Days"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted Price on Day 8:", model.predict([[8]]))
