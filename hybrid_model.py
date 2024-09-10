import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie data
movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1, 2], names=['item_id', 'title', 'genres'])

# Create a TF-IDF matrix based on genres
tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres'] = movies_df['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# Compute cosine similarity between movies based on genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return ["Movie not found"]

    # Get pairwise similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11] if i[0] != idx]

    # Return the top 10 most similar movies, excluding the input movie
    return movies_df['title'].iloc[movie_indices]


# Example recommendation
title = 'GoldenEye (1995)'
recommendations = get_movie_recommendations(title)
print(f"Movies recommended for '{title}':")
print(recommendations)
