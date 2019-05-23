import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Simple score based recommender system

metadata = pd.read_csv('./movies_metadata.csv')
print(metadata.head())
# C = mean vote across dataset
C = metadata['vote_average'].mean()
# m = number of votes of a movie in 90th quantile
# This will serve as min. score reuired for a movie to be
# considered into the top 250 list.
m = metadata['vote_count'].quantile(0.90)
# Seperating selected movies in a new dataframe
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]


# Deciding on a metric to score movie
def weighted_rating(X, m=m, C=C):
    '''
    calculate weighted ratings using formula used by imdb
    '''
    v = X['vote_count']
    R = X['vote_average']
    # IMDB formula
    return (v/(v+m)*R + (m/(v+m)*C))

# define a new feature 'score' using 'weighted_ratings()'
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
# Collecting the first 15 movies
q_movies = q_movies.sort_values('score', ascending=False)
print(q_movies.head(15))


# Content based recommender system
# We make a recommender system based on similarity with given movie
# using the description
# 1. Extract features from movie desc
# 2. Make a function that inputs a movie and outputs
#    n most similar movies

# for 1. we will use a vectorizer to get frquency of words in movie desc
# Remove words such as 'a', 'the', etc using stop_words param
vectorizer = TfidfVectorizer(stop_words='english')
# Before fitting vectorizer we remove NaN entries in 'overview' column
metadata['overview'] = metadata['overview'].fillna('')
# Fitting vectorizer
vectorizer.fit(metadata['overview'])
# Getting transformed matrrx
vec_matrix = vectorizer.transform(metadata['overview'])
# reducing size for easier computation
vec_matrix = vec_matrix[1:5000]
# Creating a similairt metric
# We will use cosine similarity (another option was Jaccard)
# Also note that since we used Tf-Idf vectorizer, our frequency matrix
# is normalized and hence using cosine similary is same as linear_kernal
# Hence we use linear_kernal as its muxh faster.
# Also note that here X and y are same -- vec_matrix
cosine_sim = linear_kernel(vec_matrix, vec_matrix)
# construct a reverse map for indices and movie titles
# to get index of movie from a given movie title
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def recommend(title, sim_mat=cosine_sim, n=10):
    # get the index of the movie that matches the title
    index = indices[title]
    # get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(sim_mat[index]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of n most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies -- getting titles
    # from indices
    return metadata['title'].iloc[movie_indices]


print(recommend('The Godfather'))
