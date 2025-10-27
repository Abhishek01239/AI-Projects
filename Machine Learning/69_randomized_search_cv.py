from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimator' : np.arrange(50,300, 50),
    'max_depth': [None, 5, 10, 15],
    'min_sample_split': [2,5,10],
    'min_sample_leaf': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=10, 
    cv = 5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best Parameters Found:")
print(random_search.best_params_)

best_model = random_search.best_estimator_
y_pred = best_model.prodict(X_test)

print("\n Test Accuracy:", accuracy_score(y_test, y_pred))