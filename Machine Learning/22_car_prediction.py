import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    "Year": [2010, 2012, 2015, 2018, 2020],
    "Mileage":[15, 14, 18, 20, 22],
    "Price": [200000, 250000, 300000, 400000, 500000]
})

X = data[["Year", "Mileage"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict([[2016, 23]], )

print("Model predction: ", y_pred)
