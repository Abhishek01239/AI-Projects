import shap
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(" Accuracy:", accuracy_score(y_test, model.predict(X_test)))

#  Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)

sample_index = 0
sample = pd.DataFrame([X_test[sample_index]], columns=feature_names)

for class_idx, class_shap_values in enumerate(shap_values):
    shap_html = shap.force_plot(
        explainer.expected_value[class_idx],
        class_shap_values[sample_index:sample_index+1, :],
        sample
    )
    filename = f"shap_force_plot_class_{class_idx}.html"
    shap.save_html(filename, shap_html)
    print(f" Saved force plot for class {class_idx} → {filename}")
