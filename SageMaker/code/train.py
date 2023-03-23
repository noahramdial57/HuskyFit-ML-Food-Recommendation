import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import json
import os

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
    
    
def train(args):
    
    files = os.listdir(args.movies)
    ratings_path = args.movies + "/ratings.csv"
    movies_path = args.movies + "/movies.csv"
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    epochs = args.epochs
    lr = args.learning_rate
    
    # Preprocess the data
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]

    # Convert movieId and userId into unique integers
    user_map = {u: i for i, u in enumerate(ratings.userId.unique())}
    ratings['user_id'] = ratings['userId'].map(user_map)

    movie_map = {m: i for i, m in enumerate(ratings.movieId.unique())}
    ratings['movie_id'] = ratings['movieId'].map(movie_map)

    # Create a matrix with users as rows and movies as columns
    matrix = torch.zeros((n_users, n_movies))
    for i, row in ratings.iterrows():
        matrix[int(row.user_id), int(row.movie_id)] = row.rating
    
    # Create model

    model = MatrixFactorization(n_users, n_foodItems)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Train the model
    for i in range(epochs):
        optimizer.zero_grad()
        user = torch.LongTensor(diningRates.user_id)
        movie = torch.LongTensor(diningRates.food_item)
        rating = torch.FloatTensor(diningRates.rating)
        predictions = model(user, movie)
        loss = criterion(predictions, rating)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(loss)
            
    test(model, ratings)
    save_model(model, args.model_dir)
    
    return
    
def test(model, ratings):
    model.eval()
    user = torch.LongTensor(ratings.user_id)
    movie = torch.LongTensor(ratings.movie_id)
    rating = torch.FloatTensor(ratings.rating)
    y_hat = model(user, movie)
    loss = F.mse_loss(y_hat, rating)
    print("test loss %.3f " % loss.item())

def save_model(model, model_dir):
    print("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    return

def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--ratings", type=str, default=os.environ["SM_CHANNEL_RATINGS"])
    parser.add_argument("--movies", type=str, default=os.environ["SM_CHANNEL_MOVIES"])

    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_args()
    train(args)