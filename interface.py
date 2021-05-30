"""
INTERFACE 
- movies with shape (#number of movies, #features(title, year, genres, ...))
- user_item_matrix with shape (#number of users, #number of movies)
- top_list with shape (#number of movies, 2)
- item-item matrix with shape (#number of popular movies, #number of popular movies)
- nmf_model: trained sklearn NMF model
"""
from tokenize import Ignore
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
from fuzzywuzzy import process


#Read all data needed for the application

movies = pd.read_csv('data/cluster_348.csv', index_col='movieid') 
movies_genres = pd.read_csv('data/movies_genres.csv', index_col='movieid') 
ratings = pd.read_csv('data/ratings.csv') 
user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col='userId')   # read in from hard-drive
titles = pd.read_csv('data/movies_rating_pivot.csv', index_col=0).columns.tolist() 
imdb = pd.read_csv('data/links.csv',dtype={'imdbId': object})
imdb = imdb.iloc[:,:2]
imdb.columns = imdb.columns.str.lower() 
cosim_titles = list(user_item_matrix.columns)



def print_movie_titles(movie_titles):
    for movie_id in movie_titles:
        print(f'> {movie_id}')
    pass


def create_user_vector(user_rating, user_item_matrix):
    """
    convert dict of user_ratings to a user_vector
    """       
    # generate the user vector
    
    user_vector = pd.Series(np.nan, index=titles)
    for movie in list(user_rating.keys()): 
        matched_title = process.extractOne(movie, titles)[0]
        user_vector[matched_title] = user_rating[movie]

    new_user_df = user_vector.to_frame().rename(columns={0:'new_user'}).T
    return new_user_df


def cosim_matrix(user_rating, user_item_matrix):
    """
    this fuction combine user vector with user_item_matrix
    Output = dataframe for cosine similarlity function

    """
    user_vector = pd.Series(np.nan, index=cosim_titles)
    for movie in list(user_rating.keys()): 
        matched_title = process.extractOne(movie, cosim_titles)[0]
        user_vector[matched_title] = user_rating[movie]
    user_df = pd.DataFrame(user_vector.transpose())  
    user_df = user_df.T      
    c_matrix = user_item_matrix.append(user_df, ignore_index = True)
    return c_matrix

def check_dup(rec_list,user_rating):
    for rec in rec_list:
        if rec not in user_rating.keys():
            return True
        else:
            return False  
