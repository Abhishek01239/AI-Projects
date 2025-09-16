import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    "Title": ["Movie1", "Movie2", "Movie3", "Movie4"],
    "Genre":["Action Adventure","Romance Drama", "Action Thriller", "Drama Comedy"]
})

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["Genre"])

similarity = cosine_similarity(genre_matrix)

def recommend(movie_title):
    idx = movies[movies["Title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key = lambda x: x[1], reverse=True)[1:]
    rec_idx = scores[0][0]
    return movies.iloc[rec_idx]["Title"]

print("If you like 'Movie1', you may also like:", recommend("Movie1")) 