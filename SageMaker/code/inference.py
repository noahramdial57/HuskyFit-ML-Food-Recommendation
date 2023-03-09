import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import json
import os
# from train import MatrixFactorization

# # Define the model
class MatrixFactorization(nn.Module):
    def __init__(self, n_users=0, n_movies=0, n_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        # initializing our matrices with a positive number generally will yield better results
        self.user_factors.weight.data.uniform_(0, 0.5)
        self.movie_factors.weight.data.uniform_(0, 0.5)
        
    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)
    
movies = pd.read_csv("/opt/ml/input/data/movies/movies.csv")
ratings = pd.read_csv("/opt/ml/input/data/movies/ratings.csv")

# Preprocess the data
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]

# Convert movieId and userId into unique integers
user_map = {u: i for i, u in enumerate(ratings.userId.unique())}
ratings['user_id'] = ratings['userId'].map(user_map)

movie_map = {m: i for i, m in enumerate(ratings.movieId.unique())}
ratings['movie_id'] = ratings['movieId'].map(movie_map)

# Make recommendations for a given user
def recommend_movies(model, user_id, num_recommendations):

    # Create a matrix with users as rows and movies as columns
    matrix = torch.zeros((n_users, n_movies))
    for i, row in ratings.iterrows():
        matrix[int(row.user_id), int(row.movie_id)] = row.rating
    with torch.no_grad():
        user = torch.LongTensor([user_map[user_id]])
        movies = torch.arange(n_movies)
        ratings = model(user, movies).detach().numpy()
    movie_ids = ratings.argsort()[-num_recommendations:][::-1]
    recommended_movies = [movies[i] for i in movie_ids]
    
    # Convert tensors to Int
    val = []
    movies = pd.read_csv('./ml-latest-small/movies.csv')
    for i in range(num_recs):
        val.append(int(recommended_movies[i]))

    L = []
    for id in val:
        row = movies.loc[movies['movieId'] == id]
        L.append(row['title'])
        
    return L

def model_fn(model_dir):
    model = MatrixFactorization(n_users, n_movies)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type =='application/json'
    data = json.loads(request_body)
    return data

def predict_fn(input_object, model):
    UserID = input_object['inputs']["UserID"]
    num_recs = input_object['inputs']["num_recs"]
    with torch.no_grad():
        prediction = recommend_movies(model, UserID, num_recs)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)

