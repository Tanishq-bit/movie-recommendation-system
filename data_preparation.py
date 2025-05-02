import pandas as pd
import numpy as np
import os
import zipfile
from urllib.request import urlretrieve

def download_movielens():
    """
    Download the MovieLens 100K dataset
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # URL for MovieLens 100K dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    # Download the dataset
    print("Downloading MovieLens dataset...")
    zip_path = os.path.join('data', 'ml-latest-small.zip')
    urlretrieve(url, zip_path)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    print("Dataset downloaded and extracted successfully!")
    
    # Return paths to the relevant files
    data_path = os.path.join('data', 'ml-latest-small')
    return {
        'movies': os.path.join(data_path, 'movies.csv'),
        'ratings': os.path.join(data_path, 'ratings.csv'),
        'links': os.path.join(data_path, 'links.csv'),
        'tags': os.path.join(data_path, 'tags.csv')
    }

def load_and_prepare_data(file_paths):
    """
    Load and prepare the MovieLens dataset for recommendation
    """
    # Load the data
    movies_df = pd.read_csv(file_paths['movies'])
    ratings_df = pd.read_csv(file_paths['ratings'])
    
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
    
    # Extract year from title and create a clean title column
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
    
    # Convert genres from pipe-separated string to list
    movies_df['genres'] = movies_df['genres'].str.split('|')
    
    # Create a pivot table for user-item ratings
    user_movie_ratings = ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    # For content-based filtering, create a genre matrix
    # One-hot encode the genres
    genres = set()
    for genre_list in movies_df['genres']:
        genres.update(genre_list)
    
    genre_matrix = pd.DataFrame(0, index=movies_df['movieId'], columns=sorted(list(genres)))
    
    for idx, row in movies_df.iterrows():
        for genre in row['genres']:
            if genre != '(no genres listed)':
                genre_matrix.loc[row['movieId'], genre] = 1
    
    return {
        'movies_df': movies_df,
        'ratings_df': ratings_df,
        'user_movie_ratings': user_movie_ratings,
        'genre_matrix': genre_matrix
    }

if __name__ == "__main__":
    file_paths = download_movielens()
    data = load_and_prepare_data(file_paths)
    
    # Print some info about the data
    print("\nMovies DataFrame:")
    print(data['movies_df'].head())
    
    print("\nRatings DataFrame:")
    print(data['ratings_df'].head())
    
    print("\nUser-Movie Ratings Matrix Shape:", data['user_movie_ratings'].shape)
    print("Genre Matrix Shape:", data['genre_matrix'].shape)