import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

X_chi2 = X_train.abs()

chi_selector = SelectKBest(score_func=chi2, k = 10)
X_chi2_selected = chi_selector.fit_transform(X_chi2, y_train)

selected_chi_features = X_train.columns[chi_selector.get_support()]
print("\n Chi-Square Selected Feature:\n", list(selected_chi_features))

anova_selector = SelectKBest(score_func=f_classif, k = 10)
X_anova_selected = anova_selector.fit_transform(X_train, y_train)

selected_anova_features = X_train.columns[anova_selector.get_support()]
print("ANOVA Selected Feature:\n", list(selected_anova_features))

model = RandomForestClassifier(random_state=42)
rfe_selector = RFE(model, n_features_to_select=10)
rfe_selector = rfe_selector.fit(X_train , y_train)

selected_rfe_features = X_train.columns[rfe_selector.support_]
print("\n RFE Selected Features:\n", list(selected_rfe_features))

model.fit(X_train[selected_rfe_features], y_train)
y_pred = model.predict(X_test[selected_rfe_features])

print("\n Accuracy using RFE-selected features: ", round(accuracy_score(y_test, y_pred),4))