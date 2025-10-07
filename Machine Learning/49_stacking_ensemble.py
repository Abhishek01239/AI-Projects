import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)), 
    ('gb', GradientBoostingClassifier(random_state=42))
]

meta_model = LogisticRegression()

stack_clf = StackingClassifier(estimators = base_models, final_estimator= meta_model)

stack_clf.fit(X_train, y_train)

y_pred = stack_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Stacking Model Accuracy: {acc*100:.2f}%")

