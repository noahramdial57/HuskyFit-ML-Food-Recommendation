import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import json
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
import boto3

start = time.time()


# Define the model
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
    
# Load the  dataset
# bucket = "foodrecdata"
# s3 = boto3.client('s3')

# key = 'normalizedValues.csv'
# response = s3.get_object(Bucket=bucket, Key=key)
# users_file = pd.read_csv(response['Body'])

# key = 'dining_ratings.csv'
# response2 = s3.get_object(Bucket=bucket, Key=key)
# df2 = pd.read_csv(response2['Body'])

# key = 'alldininghalls.csv'
# response3 = s3.get_object(Bucket=bucket, Key=key)
# df3 = pd.read_csv(response3['Body'])

# users = users_file.copy() # normalizedValues.csv
# diningRates = df2.copy()      # dining_ratings.csv
# diningHalls = df3.copy()           # alldininghalls.cs

users       = pd.read_csv("./preprocessed-data/normalizedValues.csv") # normalizedValues.csv
diningRates = pd.read_csv("./preprocessed-data/dining_ratings.csv")   # dining_ratings.csv
diningHalls = pd.read_csv("./preprocessed-data/alldininghalls.csv")    # alldininghalls.cs


# Preprocess the data
# print(diningRates)
n_users = diningRates.userId.unique().shape[0]
n_foodItems = diningRates.foodItem.unique().shape[0]

# Convert movieId and userId into unique integers
user_map = {u: i for i, u in enumerate(diningRates.userId.unique())}
diningRates['user_id'] = diningRates['userId'].map(user_map)

dining_map = {m: i for i, m in enumerate(diningRates.foodItem.unique())}
diningRates['food_item'] = diningRates['foodItem'].map(dining_map)

# Create a matrix with users as rows and food as columns
# matrix = torch.zeros((n_users, n_foodItems))
# for i, row in diningRates.iterrows():
#     matrix[int(row.user_id), int(row.food_item)] = row.rating

def model_fn(model_dir):
    model = MatrixFactorization(n_users, n_foodItems)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    data = json.loads(request_body)
    return data

def predict_fn(input_object, model):
    UserID = input_object['inputs']["UserID"]
    # num_recs = input_object['inputs']["num_recs"]
    dHallPref = input_object['inputs']["Dining Hall Preference"]
    allergens = input_object['inputs']["Allergens"]
    dietary_restrictions = input_object['inputs']["Dietary Restrictions"]
    meal = input_object['inputs']["Meal"]

    df = getSimilarUserRecs(model, UserID, 10) # dataframe
    recs = contentFiltering(df, dHallPref, allergens, dietary_restrictions, meal)["Food Item"].tolist()

    return recs

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    # res = predictions.cpu().numpy().tolist()
    return json.dumps(predictions)

# -------------------------------------------------------------------------------------------------

# Make recommendations for a given user
def recommend_food(model, user_id, num_recommendations):
    with torch.no_grad():
        user = torch.LongTensor([user_map[user_id]])
        food_items = torch.arange(n_foodItems)
        ratings = model(user, food_items).detach().numpy()
    food_ids = ratings.argsort()[-num_recommendations:][::-1]
    recommended_food = [food_items[i] for i in food_ids]
    return recommended_food

# Get recommendations for a user with user_id 1
def getRecs(model, user_id, num_recs):
    recommended_food = recommend_food(model, user_id, num_recs)

    # Convert tensors to Int
    val = []
    for i in range(num_recs):
        val.append(int(recommended_food[i]))

    for id in val:
        row = diningHalls.loc[diningHalls['foodId'] == id]
        movie = row.values.tolist()
        if len(movie) == 0:
            continue
    return val

# recommendation for that particular user:

def getSimilarUserRecs(model, userID, num_recs = 5):
    recs = []
    recs += getRecs(model, userID, num_recs)
    # recommendation for similar users:
    
    similar_users = find_similar_users(userID) # added this
    for user in similar_users['UserID'].tolist():
        recs += getRecs(model, user, 1)
    recs = set(recs)
    food_recs = diningHalls[diningHalls['foodId'].isin(recs)]
        
    return food_recs

# Now do content filtering

def contentFiltering(df, dHall_pref, allergens, diet_restr, meal):     
        
    # filter by meal
    recs = df.loc[(df[meal] == 1)] # works
    recs = recs.drop_duplicates(subset="Food Item") # remove duplicates
        
    # filter by dining 
    if len(dHall_pref) == 0:
        pass
    else:
        for index, row in recs.iterrows():
            L = []
            actual_index = row["foodId"]
            for pref in dHall_pref:  

                if row[pref] == 0: L.append(0)
                else: L.append(1)

            all_zeros = all(val == 0 for val in L)
            if all_zeros == True: 
                recs = recs.drop(actual_index)
            
    # filter by allergens 
    if len(allergens) == 0:
        pass
    else:
        for index, row in recs.iterrows():
            L = []
            actual_index = row["foodId"]
            for allergen in allergens:  

                if row[allergen] == 1: L.append(1)
                else: L.append(0)

            all_zeros = all(val == 0 for val in L)
            # If there is an allergen present, a 1 in the list, drop it
            if all_zeros == False: 
                recs = recs.drop(actual_index)

    # filter by dietary restrictions
    if len(diet_restr) == 0:
        pass
    else:
        for index, row in recs.iterrows():
            L = []
            actual_index = row["foodId"]
            for restr in diet_restr:  

                if row[restr] == 0: L.append(0)
                else: L.append(1)

            all_zeros = all(val == 0 for val in L)
            # If there is an allergen present, a 1 in the list, drop it
            if all_zeros == True: 
                recs = recs.drop(actual_index)
                     
    return recs

def find_similar_users(userID, k=25):    
    
    users_df = users.copy()
    # one_user = users[:1].copy() # new user data frame
    new_user_df = users.iloc[userID].to_frame().T # how to access a specific row
    
    # Compute cosine similarity between new user and existing users
    new_user_df.drop(columns = ['UserID'])
    users_df.drop(columns = ['UserID'])
    similarities = cosine_similarity(new_user_df, users_df)[0]
    
    # Find the top-k similar users
    top_k_similar_users_indices = similarities.argsort()[-k-1:-1][::-1]
    top_k_similar_users = users.iloc[top_k_similar_users_indices]
    
    return top_k_similar_users

end = time.time()
print(end - start)
