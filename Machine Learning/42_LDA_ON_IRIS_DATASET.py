import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

lda_df = pd.DataFrame({
    "LD1": X_lda[:,0],
    "LD2":X_lda[:,1],
    'target':y
})

plt.figure(figsize=(8,6))
for i, target in enumerate(np.unique(y)):
    subset = lda_df[lda_df['target'] == target]
    plt.scatter(subset['LD1'], subset['LD2'], label = target_names[i])

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA - Iris Dataset (2D Visualization)")
plt.legend()
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Accuracy with LDA + Logistic Regression:", accuracy_score(y_test, y_pred))
