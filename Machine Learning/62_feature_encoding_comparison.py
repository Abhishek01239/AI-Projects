import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder

data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Green', 'Blue', 'Red', 'Blue'],
    'Size': ['S', 'M', 'L', 'L', 'M', 'S', 'M', 'L'],
    'Bought': [1, 0, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print(" Original Dataset:\n", df)

X = df[['Color', 'Size']]
y = df['Bought']

label_enc = LabelEncoder()
X_label = X.copy()
for col in X_label.columns:
    X_label[col] = label_enc.fit_transform(X_label[col])

X_train, X_test, y_train, y_test = train_test_split(X_label, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\n Label Encoding Accuracy: {accuracy_score(y_test, y_pred):.4f}")

onehot = OneHotEncoder(sparse_output=False, drop=None)
X_onehot = onehot.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f" One-Hot Encoding Accuracy: {accuracy_score(y_test, y_pred):.4f}")

target_enc = TargetEncoder()
X_target = target_enc.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_target, y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f" Target Encoding Accuracy: {accuracy_score(y_test, y_pred):.4f}")
