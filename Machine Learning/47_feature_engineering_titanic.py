import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

titanic = sns.load_dataset("titanic").dropna(subset=["age", "fare","embarked", "sex","class", "survived"])

df = titanic.copy()

df["sex"] = df["sex"].map({"male":0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C":1,"Q":2})
df["class"] = df["class"].map({"Third": 3, "Second": 2, "First": 1})

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]

X = df[["pclass","sex", "age", "fare_per_person","embarked", "class", "family_size", "is_alone"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index = X.columns).sort_values(ascending= False)
print("\nFeature Importance:\n", importance)