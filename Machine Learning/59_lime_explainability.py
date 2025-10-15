import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular

data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators= 100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names = feature_names, 
    class_names=[str(c) for c in sorted(set(y))],
    mode = "classification"
)

sample_index = 0
sample = X_test[sample_index]

exp = explainer.explain_instance(
    sample, 
    model.predict_proba,
    num_features = len(feature_names)
)

exp.save_to_file("lime_explanation.html")
print("LIME explanation saved -> lime_explanation.html")