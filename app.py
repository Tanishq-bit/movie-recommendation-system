from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd
import numpy as np
import json
import traceback

from data_preparation import download_movielens, load_and_prepare_data
from recommendation_models import CollaborativeFiltering, ContentBasedFiltering, HybridRecommender

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Use our custom encoder for all jsonify calls

# Global variables to store our data and models
movies_df = None
ratings_df = None
user_movie_ratings = None
cf_model = None
cb_model = None
hybrid_model = None

# Force retraining flag - set to True to ignore existing models
FORCE_RETRAINING = False


def load_or_train_models():
    """
    Load models from disk if they exist, otherwise train new ones
    """
    global movies_df, ratings_df, user_movie_ratings, cf_model, cb_model, hybrid_model
    
    # Skip loading if force retraining is enabled
    if FORCE_RETRAINING:
        print("Force retraining enabled. Skipping model loading...")
    # Check if we have saved models
    elif os.path.exists('models/hybrid_model.pkl'):
        print("Loading models from disk...")
        try:
            # Try to load each model separately with error handling
            try:
                with open('models/movies_df.pkl', 'rb') as f:
                    movies_df = pickle.load(f)
                print("Successfully loaded movies_df")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading movies_df: {str(e)}")
                return False
                
            try:
                with open('models/ratings_df.pkl', 'rb') as f:
                    ratings_df = pickle.load(f)
                print("Successfully loaded ratings_df")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading ratings_df: {str(e)}")
                return False
                
            try:
                with open('models/user_movie_ratings.pkl', 'rb') as f:
                    user_movie_ratings = pickle.load(f)
                print("Successfully loaded user_movie_ratings")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading user_movie_ratings: {str(e)}")
                return False
            
            try:
                with open('models/cf_model.pkl', 'rb') as f:
                    cf_model = pickle.load(f)
                print("Successfully loaded cf_model")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading cf_model: {str(e)}")
                return False
            
            try:
                with open('models/cb_model.pkl', 'rb') as f:
                    cb_model = pickle.load(f)
                print("Successfully loaded cb_model")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading cb_model: {str(e)}")
                return False
            
            try:
                with open('models/hybrid_model.pkl', 'rb') as f:
                    hybrid_model = pickle.load(f)
                print("Successfully loaded hybrid_model")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading hybrid_model: {str(e)}")
                return False
                
            print("All models loaded successfully!")
            return True
        except Exception as e:
            print(f"Unexpected error during model loading: {str(e)}")
            print(traceback.format_exc())
            return False
    
    # Download and prepare data
    print("Downloading and preparing data...")
    try:
        file_paths = download_movielens()
        data_dict = load_and_prepare_data(file_paths)
        
        movies_df = data_dict['movies_df']
        ratings_df = data_dict['ratings_df']
        user_movie_ratings = data_dict['user_movie_ratings']
        genre_matrix = data_dict['genre_matrix']
        
        # Train models
        print("Training collaborative filtering model...")
        cf_model = CollaborativeFiltering(user_movie_ratings)
        cf_model.compute_user_similarity()
        cf_model.compute_item_similarity()
        
        print("Training content-based filtering model...")
        cb_model = ContentBasedFiltering(movies_df, genre_matrix)
        
        print("Setting up hybrid recommender...")
        hybrid_model = HybridRecommender(cf_model, cb_model, movies_df)
        
        # Save models
        if not os.path.exists('models'):
            os.makedirs('models')
            
        print("Saving models to disk...")
        try:
            # Save each model with error handling
            with open('models/movies_df.pkl', 'wb') as f:
                pickle.dump(movies_df, f)
            
            with open('models/ratings_df.pkl', 'wb') as f:
                pickle.dump(ratings_df, f)
            
            with open('models/user_movie_ratings.pkl', 'wb') as f:
                pickle.dump(user_movie_ratings, f)
            
            with open('models/cf_model.pkl', 'wb') as f:
                pickle.dump(cf_model, f)
            
            with open('models/cb_model.pkl', 'wb') as f:
                pickle.dump(cb_model, f)
            
            with open('models/hybrid_model.pkl', 'wb') as f:
                pickle.dump(hybrid_model, f)
            
            print("Models saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            print(traceback.format_exc())
            # Continue without saving
            return True
    except Exception as e:
        print(f"Error during data preparation or model training: {str(e)}")
        print(traceback.format_exc())
        raise


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types to make them JSON serializable
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


@app.route('/')
def index():
    """
    Render the home page
    """
    # Get a list of all genres
    genres = set()
    for genre_list in movies_df['genres']:
        genres.update(genre_list)
    genres = sorted(list(genres))
    
    # Get a list of all users for the demo
    users = sorted(user_movie_ratings.index.tolist())
    
    return render_template('index.html', genres=genres, users=users)


@app.route('/recommend_by_user', methods=['POST'])
def recommend_by_user():
    """
    Generate recommendations for a user
    """
    user_id = int(request.form.get('user_id'))
    num_recommendations = int(request.form.get('num_recommendations', 10))
    
    try:
        # Get recommendations
        movie_ids = hybrid_model.recommend_for_user(user_id, n=num_recommendations)
        movies = hybrid_model.get_movie_details(movie_ids)
        
        # Convert any NumPy types to ensure JSON serialization works
        movies = convert_numpy_types(movies)
        
        return jsonify({
            'success': True,
            'recommendations': movies
        })
    except Exception as e:
        print(f"Error in recommend_by_user: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/recommend_by_genres', methods=['POST'])
def recommend_by_genres():
    """
    Generate recommendations based on selected genres
    """
    selected_genres = request.form.getlist('genres[]')
    num_recommendations = int(request.form.get('num_recommendations', 10))
    
    try:
        # Get recommendations
        movie_ids = hybrid_model.recommend_by_genres(selected_genres, n=num_recommendations)
        movies = hybrid_model.get_movie_details(movie_ids)
        
        # Convert any NumPy types to ensure JSON serialization works
        movies = convert_numpy_types(movies)
        
        return jsonify({
            'success': True,
            'recommendations': movies
        })
    except Exception as e:
        print(f"Error in recommend_by_genres: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/rate_movie', methods=['POST'])
def rate_movie():
    """
    Save a user's rating for a movie
    """
    user_id = int(request.form.get('user_id'))
    movie_id = int(request.form.get('movie_id'))
    rating = float(request.form.get('rating'))
    
    # In a real application, you would save this to a database
    # For this demo, we'll update our in-memory data
    try:
        # Check if user exists
        if user_id not in user_movie_ratings.index:
            # Create a new user
            new_ratings = pd.Series(0, index=user_movie_ratings.columns)
            user_movie_ratings.loc[user_id] = new_ratings
            
        # Update the rating
        user_movie_ratings.loc[user_id, movie_id] = rating
        
        # Add to the ratings dataframe
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating],
            'timestamp': [int(pd.Timestamp.now().timestamp())]
        })
        global ratings_df
        ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
        
        # In a real application, you would retrain your models periodically
        # For this demo, we'll retrain immediately
        global cf_model
        cf_model.compute_user_similarity()
        
        return jsonify({
            'success': True,
            'message': f'Rating of {rating} saved for movie {movie_id} by user {user_id}'
        })
    except Exception as e:
        print(f"Error in rate_movie: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/search_movies', methods=['GET'])
def search_movies():
    """
    Search for movies by title
    """
    query = request.args.get('query', '').lower()
    
    if not query:
        return jsonify({
            'success': True,
            'movies': []
        })
    
    # Search for movies that contain the query in their title
    matching_movies = movies_df[movies_df['clean_title'].str.lower().str.contains(query)]
    
    # Convert to a list of dictionaries
    movies = []
    for _, movie in matching_movies.head(10).iterrows():
        # Convert NumPy types to Python native types
        movie_dict = {
            'id': int(movie['movieId']),
            'title': str(movie['title']),
            'genres': [str(g) for g in movie['genres']],
            'year': int(movie['year']) if not pd.isna(movie['year']) else None
        }
        movies.append(movie_dict)
    
    return jsonify({
        'success': True,
        'movies': movies
    })


if __name__ == '__main__':
    # Set this to True to force retraining
    FORCE_RETRAINING = True
    
    # Load or train models
    success = load_or_train_models()
    if not success:
        print("WARNING: There was an issue with model loading. Setting FORCE_RETRAINING to True and trying again.")
        FORCE_RETRAINING = True
        load_or_train_models()
    
    # Run the Flask app
    app.run(debug=True)