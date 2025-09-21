# Project 45 - LightGBM Classifier (Fraud Detection)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample dataset
data = {
    'TransactionAmount': [200, 5000, 150, 12000, 300, 7500, 100, 9000, 400, 8500],
    'AccountAge': [2, 10, 1, 12, 3, 15, 2, 11, 4, 13],
    'LocationRisk': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 1 = risky area
    'DeviceRisk': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    'Fraud': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# 2. Features and target
X = df[['TransactionAmount', 'AccountAge', 'LocationRisk', 'DeviceRisk']]
y = df['Fraud']

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. LightGBM Classifier
model = LGBMClassifier()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
