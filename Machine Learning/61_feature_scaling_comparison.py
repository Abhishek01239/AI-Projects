import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = {
    "StandartScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "Normalizer": Normalizer()

}

result = []

for name, scaler in scaler.items():

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    result.append((name, acc))
    print(f"{name} Accuracy: {acc: .4f}")

df = pd.DataFrame(result, columns = ["Scaler", "Accuracy"]).sort_values(by = "Accuracy", ascending=False)
print("\n Scaling Method Comparison:\n", df)
df.to_csv("scaling_comparison_result.csv", index =False)
print("Reslutd saved -> scaling_comparison_results.csv")