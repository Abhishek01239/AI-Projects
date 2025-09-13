import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    "Size": [1000, 1500, 2000, 2500, 3000],
    "Price": [100000,150000, 200000, 250000, 300000]
})

X = data[["Size"]]
y = data[["Price"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Prediction for 2200 sqft:", model.predict([[3430]]))