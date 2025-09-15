from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X,y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Prediction for first tesst digit:", model.predict([X_test[0]]))
print("Actual:", y_test[0])