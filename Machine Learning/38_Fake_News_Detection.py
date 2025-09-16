from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

news = [
    "Government announces new scheme", 
    "Breaking: Aliens landed in my backyard",
    "Stock market reaches record high",
    "Click here to win free money"
]
labels = [0, 1, 0, 1]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(news)

X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Pridiction:", model.predict(vectorizer.transform(["Earth is going to blast"])))
