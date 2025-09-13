from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X,y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)

print("Prediction for first test sample: ", model.predict(X_test[0]))
print("Actual:" , y_test[0])