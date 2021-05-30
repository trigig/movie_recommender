"""
Contains various recommondation implementations
all algorithms return a list of tuple (title,movieid)
"""

from logging import error
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from interface import movies, user_item_matrix, create_user_vector, movies_genres, titles, cosim_matrix, imdb,check_dup
from fuzzywuzzy import process
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def select_random_movies(movies, user_rating, k = 5):
    '''
    select a random movie by sampling a random number from all movies
    check if title is in the user rating dict, 
    then return a list of tuple (title,movieid)

    '''
    rec = movies['title'].sample(n = k)
    for title in rec:
        if title in user_rating.keys():
            rec = movies['title'].sample(n = k)
        else:
            pass            
    rec = pd.DataFrame(rec)
    movieid = pd.merge(rec,imdb, how='left', on ='movieid')
    random_recs =[(title,id,imdbid) for title, id, imdbid in zip(movieid['title'],movieid['movieid'],movieid['imdbid'])]
    return random_recs



def recommend_from_same_cluster(user_rating, movies, k=5):
    """
    Return k most similar movies to the one spicified in the movieID
    
    INPUT
    - user_rating: a dictionary of titles and ratings
    - movies: a data frame with movie titles and cluster number
    - k: number of movies to recommend

    OUTPUT
    - a list of tuple (title,movieid)
    """
    user_df = pd.DataFrame({'title':list(user_rating.keys()), 'rating':list(user_rating.values())})
    favourite_movie = user_df.loc[user_df['rating'] == user_df['rating'].max(), 'title'].sample()
    favourite_movie_title = process.extractOne(favourite_movie.iloc[0], movies['title'])[0]
    cluster = movies.loc[movies['title']==favourite_movie_title, 'cluster'].iloc[0]
    movie_titles = (movies.loc[movies['cluster']==cluster, 'title'].sample(n=k))
    rec_list = list(movie_titles.values)
    check = check_dup(rec_list,user_rating)
    if check == False:
        movie_titles = (movies.loc[movies['cluster']==cluster, 'title'].sample(n=k))
        rec_list = list(movie_titles.values)
    elif check == True:
        movie_titles = pd.DataFrame(movie_titles)
        movieid = pd.merge(movie_titles,imdb, how='left', on ='movieid')
        movie_rec =[(title,id,imdbid) for title, id, imdbid in zip(movieid['title'],movieid['movieid'],movieid['imdbid'])]
    return movie_rec

def recommend_with_NMF(user_rating,user_item_matrix, k=5):
    """
    NMF Recommender
    1) Take user input and convert to dataframe with all titles
    2) Fill na 
    3) read trained NMF model
    4) Output as a list of tuple (title,movieid)
    
    """
    
    raw_user_vec = create_user_vector(user_rating, user_item_matrix)
    new_user_vec =raw_user_vec.fillna(2.5)
    pkl_filename = 'data/NMF_Model_50_components.pkl'
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    Q =model.components_
    R_user= model.transform(new_user_vec)
    rec = np.dot(R_user,Q)
    rec= rec.reshape(9433,)
    raw_user_vec = raw_user_vec.values
    raw_user_vec = raw_user_vec.reshape(9433,)
    rec = pd.DataFrame({'user_input':raw_user_vec, 'predicted_ratings':rec}, index = titles)
    rec = rec[rec['user_input'].isna()].sort_values(by = 'predicted_ratings', ascending= False).head(k)
    rec= list(rec.index)
    rec = pd.DataFrame(rec, columns=['title'])
    movies_copy = movies.reset_index()
    movies_copy = movies_copy.iloc[:,:2]
    movieid = pd.merge(rec, movies_copy, how='left', on ='title')
    movieid = pd.merge(movieid,imdb, how='left', on ='movieid')
    nmf_rec = [(title,id,imdbid) for title, id, imdbid in zip(movieid['title'],movieid['movieid'],movieid['imdbid'])]
    return nmf_rec    
   

def recommend_with_user_similarity(user_rating, user_item_matrix, k=5):
    df_raw = cosim_matrix(user_rating, user_item_matrix)
    df = df_raw.fillna(0)
    user_cosim = cosine_similarity(df)
    user = user_cosim[610,:-1]
    nearest_user = np.argmax(user)
    recs = df_raw.iloc[nearest_user,:].dropna().sort_values(ascending=False)
    recs =list(recs.index)
    recs = [title for title in recs if title not in user_rating.keys()]
    recs = recs[:k]
    recs = pd.DataFrame(recs, columns=['title'])
    movies_copy = movies.reset_index()
    movies_copy = movies_copy.iloc[:,:2]
    movieid = pd.merge(recs, movies_copy, how='left', on ='title')
    movieid = pd.merge(movieid,imdb, how='left', on ='movieid')
    cosim_rec = [(title,id) for title, id in zip(movieid['title'],movieid['movieid'])]
    cosim_rec = [(title,id,imdbid) for title, id, imdbid in zip(movieid['title'],movieid['movieid'],movieid['imdbid'])]    
    return cosim_rec
