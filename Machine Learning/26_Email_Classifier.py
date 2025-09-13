from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Win money now for free", "Limited offer, claim prize for free",
    "Meeting at 5pm", "Lunch tomorrow?",
    "Get cheap loans for free", "Project deadline extended"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Prediction:", model.predict(vectorizer.transform(["Win a free iPhone now"])))
