from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this movie", "This is terrible", "Fantastic acting","Worst film ever", "So good", "I hate this"]
labels = [1,0,1,0,1,0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state= 42)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Prediction:",model.predict(vectorizer.transform(["the movie was good and with good acting"])))