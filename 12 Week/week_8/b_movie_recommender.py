# Building a Movie Recommender
# starting on page 477
import pandas as pd

df = pd.read_csv('./data/tmdb_5000_movies.csv')
df.info()
print('Columns\n', df.columns, '\n')

df = df[['title', 'tagline', 'overview', 'genres', 'popularity']]
df.tagline.fillna('', inplace=True)
df['description'] = df['tagline'].map(str) + ' ' + df['overview']
df.dropna(inplace=True)
df.info()
print('\nSimplified DF:\n', df.head(), '\n')

# Text preprocessing - starting on page 480
import nltk
import numpy as np
import re

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special chars/whitespace
    doc = re.sub('[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize doc
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

df['description'] = \
        df['description'].apply(lambda x: normalize_corpus(x))
norm_corpus = df['description']
print('Length of normalized corpus:', len(norm_corpus), '\n')
print(df.info(), '\n')

# Save this updated corpus df
df.to_csv('./data/norm_corpus.csv')

# Extract TF-IDF Features - page 481
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
print('TFIDF matrix shape:\n', tfidf_matrix.shape, '\n')

# Cosine similarity and pairwise doc similarity page 482
from sklearn.metrics.pairwise import cosine_similarity
doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)
print('Document similarity df:\n', doc_sim_df.head(),'\n')

# Movie list page 482
movies_list = df['title'].values
print('Movies list:\n', movies_list, movies_list.shape, '\n')

# Find top similar movies for a sample movie page 483
movie_idx = np.where(movies_list == 'Minions')[0][0]
print('Movie like Minions:\n', movie_idx, '\n')

# Movie similarities
movie_similarities = doc_sim_df.iloc[movie_idx].values
print('Movie similarities, like Minions:\n', movie_similarities, '\n')

# Top five similar movie IDs
similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
print('Similar movie indices like Minions:\n', similar_movie_idxs, '\n')

# Get top five similar movies page 484
similar_movies = movies_list[similar_movie_idxs]
print('Similar movies to Minions:\n', similar_movies, '\n')

# Build a movie recommender page 484
def movie_recommender(movie_title, movies=movies_list, doc_sims=doc_sim_df):
    # find movie id
    movie_idx = np.where(movies == movie_title)[0][0]
    # get movie similarities
    movie_similarities = doc_sims.iloc[movie_idx].values
    # get top 5 similar movie ids
    similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
    # get top 5 movies
    similar_movies = movies[similar_movie_idxs]
    # return the top 5 movies
    return similar_movies

popular_movies = df.sort_values(by='popularity', ascending=False)
print('Popular movies:\n', popular_movies, '\n')

# Just 5 movies
for movie in popular_movies['title'][0:5]:
    print('Movie:', movie)
    print('Top 5 recommended movies:', movie_recommender(movie_title=movie), '\n')


