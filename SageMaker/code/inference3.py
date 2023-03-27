import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import boto3
from datetime import date

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
bucket = "foodrecdata"
s3 = boto3.client('s3')

key = 'normalizedValues.csv'
response = s3.get_object(Bucket=bucket, Key=key)
users_file = pd.read_csv(response['Body'])

key = 'dining_ratings.csv'
response2 = s3.get_object(Bucket=bucket, Key=key)
df2 = pd.read_csv(response2['Body'])

key = 'alldininghalls.csv'
response3 = s3.get_object(Bucket=bucket, Key=key)
df3 = pd.read_csv(response3['Body'])

users = users_file.copy() # normalizedValues.csv
diningRates = df2.copy()      # dining_ratings.csv
diningHalls = df3.copy()           # alldininghalls.cs

other = ["UserID", "Weight (lbs)", "Height"]
meals = ['Breakfast', 'Lunch', 'Dinner']
allergens = ['Fish', 'Soybeans', 'Wheat', 'Gluten', 'Milk', 'Tree Nuts', 'Eggs', 'Sesame', 'Crustacean Shellfish']
dietary_restrictions = ['Gluten Friendly', 'Less Sodium', 'Smart Check', 'Vegan', 'Vegetarian', 'Contains Nuts']
dHalls = ["gelfenbien", "kosher", "north", "northwest", "McMahon", "putnam", "south", "whitney"]

# users_file = os.path.abspath("/home/ec2-user/SageMaker/preprocessed-data/normalizedValues.csv")
# diningRates_file = os.path.abspath("/home/ec2-user/SageMaker/preprocessed-data/dining_ratings.csv")
# diningHalls_file = os.path.abspath("/home/ec2-user/SageMaker/preprocessed-data/alldininghalls.csv")

# users = pd.read_csv(users_file)
# diningRates = pd.read_csv(diningRates_file)
# diningHalls = pd.read_csv(diningHalls_file)

# Preprocess the data
n_users = diningRates.userId.unique().shape[0]
n_foodItems = diningRates.foodItem.unique().shape[0]

# Convert movieId and userId into unique integers
user_map = {u: i for i, u in enumerate(diningRates.userId.unique())}
diningRates['user_id'] = diningRates['userId'].map(user_map)

dining_map = {m: i for i, m in enumerate(diningRates.foodItem.unique())}
diningRates['food_item'] = diningRates['foodItem'].map(dining_map)

def model_fn(model_dir):
    model = MatrixFactorization(n_users, n_foodItems)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to('cpu').eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/json'
    data = json.loads(request_body)
    return data

def predict_fn(input_object, model):
    UserID = int(input_object['inputs']["UserID"])
    # num_recs = input_object['inputs']["num_recs"]
    dHallPref = input_object['inputs']["Dining Hall Preference"]
    allergens = input_object['inputs']["Allergens"]
    dietary_restrictions = input_object['inputs']["Dietary Restrictions"]
    meal = input_object['inputs']["Meal"]
    height = input_object['inputs']["Height"]
    weight = input_object['inputs']["Weight"]
    
    user = constructDataframe(UserID, dHallPref, allergens, dietary_restrictions, meal, height, weight)


    df = getSimilarUserRecs(model, user, 10) # dataframe
    recs = contentFiltering(df, dHallPref, allergens, dietary_restrictions, meal)
    check = checkDiningHall(recs, dHallPref, meal)

    return check

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

def getSimilarUserRecs(model, userID, num_recs = 5):
    similar_users = find_similar_users(userID) # added this
    
    replacement = similar_users['UserID'].values[:1][0]
    replacement_user = similar_users.tail(-1) # remove first row

    recs = []
    recs += getRecs(model, replacement, num_recs)
    
    # recommendation for similar users:
    for user in replacement_user['UserID'].tolist():
        recs += getRecs(model, user, num_recs)
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
                     
    df_input = ["foodId", "Food Item"]
    for pref in dHall_pref:
        df_input.append(pref)

    a = recs[df_input] # gets only necessary values
    obj = a.to_dict(orient="records") 
        
    return obj

def find_similar_users(user, k=25):    
    
    users_df = users.copy()
    new_user_df = user
        
    # Compute cosine similarity between new user and existing users
    new_user_df.drop(columns = ['UserID'])
    users_df.drop(columns = ['UserID'])
    similarities = cosine_similarity(new_user_df, users_df)[0]
    
    # Find the top-k similar users
    top_k_similar_users_indices = similarities.argsort()[-k-1:-1][::-1]
    top_k_similar_users = users.iloc[top_k_similar_users_indices]
    
    # display(top_k_similar_users)
    
    return top_k_similar_users

# Check to see if food is being served at specified Dining hall
def checkDiningHall(recs, dHallPref, meal):
    bucket = "dininghall-data-cache"

    s3 = boto3.client('s3')
    today = date.today()
    dd = today.strftime("%d")
    mm = today.strftime("%m")
    yyyy = today.strftime("%Y")
     
    L = []
    # Check to see if user has a preference
    if len(dHallPref) > 0:
        
        # For each recommended item
        for item in recs:
        
            # Check all preferred dining halls to see if food item is being served
            for pref in dHallPref:

                # If a user has a preference, see if its currently being served at their preferred dining hall
                if item[pref] == 1:
                    dHall = item[pref]

                    key = "{}-{}-{}-{}-{}.json".format(pref.capitalize(), meal, mm, dd, yyyy) # pref is the dining hall
                    response = s3.get_object(Bucket=bucket, Key=key)
                    content = response['Body'].read()
                    data = json.loads(content)

                    for food_item in data:
                        if food_item["Food Item"] == item["Food Item"]:
                            # print("IT ACTUALLY FUCKING WORKS!!!")
                            L.append(food_item)
                            
    else:
        # User does not have a preference, recommend food from anywhere
        dHalls = ["gelfenbien", "kosher", "north", "northwest", "McMahon", "putnam", "south", "whitney"]

        for item in recs:
            
            # Check each dining hall to see if item is being served
            for dining_hall in dHalls:

                if dining_hall == "McMahon": continue
                else: dining_hall = dining_hall.capitalize()


                key = "{}-{}-{}-{}-{}.json".format(dining_hall, meal, mm, dd, yyyy) # pref is the dining hall
                response = s3.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()
                data = json.loads(content)

                for food_item in data:
                    if food_item["Food Item"] == item["Food Item"]:
                        # print("IT ACTUALLY FUCKING WORKS!!!")
                        L.append(food_item)          

    # Make sure we dont return too many recommendations
    if len(L) > 5:
        L = L[0:5]
        
    return L      

def constructDataframe(userID, user_dHallPref, user_allergens, user_dietary_restrictions, user_meal, user_height, user_weight):
    # Construct User Dataframe
    features = other + allergens + dietary_restrictions + dHalls
    L = []

    L.append(userID)
    L.append(user_weight)
    L.append(user_height)


    for allergen in allergens:

        if allergen in user_allergens: L.append(1)
        else: L.append(0)

    for diet_restr in dietary_restrictions:

        if diet_restr in user_dietary_restrictions: L.append(1)
        else: L.append(0)

    for dHall in dHalls:

        if dHall in user_dHallPref: L.append(1)
        else: L.append(0)


    # user dataframe
    user = pd.DataFrame([L], columns = features, index=['610'])

    # Normalize height and weight values
    user['Height'] = user['Height'].apply(lambda x:int(x.split('\'')[0])*12+int(x.split('\'')[1]))
    numerical_features = ['Weight (lbs)', 'Height']
    numerical_data = user[numerical_features]
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)
    user[numerical_features] = numerical_data_scaled

    return user

