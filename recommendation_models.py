import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class CollaborativeFiltering:
    """
    Implements user-based and item-based collaborative filtering
    """
    def __init__(self, user_movie_ratings):
        """
        Initialize with user-movie ratings matrix
        
        Parameters:
        user_movie_ratings (pandas.DataFrame): Pivot table with users as rows, movies as columns, ratings as values
        """
        self.user_movie_ratings = user_movie_ratings
        self.user_similarity = None
        self.item_similarity = None
    
    def compute_user_similarity(self):
        """
        Compute similarity between users using cosine similarity
        """
        # Transpose the matrix to compute similarity between users
        self.user_similarity = cosine_similarity(self.user_movie_ratings)
        return self.user_similarity
    
    def compute_item_similarity(self):
        """
        Compute similarity between items (movies) using cosine similarity
        """
        # Compute similarity between movies
        self.item_similarity = cosine_similarity(self.user_movie_ratings.T)
        return self.item_similarity
    
    def user_based_recommendation(self, user_id, n=10):
        """
        Generate recommendations for a user based on similar users' preferences
        
        Parameters:
        user_id (int): The user ID to generate recommendations for
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs recommended for the user
        """
        if self.user_similarity is None:
            self.compute_user_similarity()
        
        # Get the user's index
        user_idx = self.user_movie_ratings.index.get_loc(user_id)
        
        # Get similarity scores for this user with all other users
        user_similarities = self.user_similarity[user_idx]
        
        # Get movies the user has already rated
        user_rated_movies = self.user_movie_ratings.iloc[user_idx].to_numpy().nonzero()[0]
        
        # Calculate weighted average of all users' ratings
        weighted_ratings = np.zeros(self.user_movie_ratings.shape[1])
        similarity_sum = np.zeros(self.user_movie_ratings.shape[1])
        
        for other_user_idx in range(len(self.user_movie_ratings)):
            # Skip the user themselves
            if other_user_idx == user_idx:
                continue
            
            # Get the similarity with this other user
            sim = user_similarities[other_user_idx]
            
            # Skip if similarity is too low (optional)
            if sim <= 0:
                continue
            
            # Get ratings from this other user
            other_user_ratings = self.user_movie_ratings.iloc[other_user_idx].values
            
            # Add weighted ratings
            weighted_ratings += sim * other_user_ratings
            similarity_sum += np.where(other_user_ratings > 0, sim, 0)
        
        # Avoid division by zero
        similarity_sum = np.where(similarity_sum == 0, 1, similarity_sum)
        
        # Get the final predicted ratings
        predicted_ratings = weighted_ratings / similarity_sum
        
        # Set already rated movies to have a negative rating so they won't be recommended
        predicted_ratings[user_rated_movies] = -1
        
        # Get top N recommendations
        recommended_indices = predicted_ratings.argsort()[-n:][::-1]
        
        # Convert indices to movie IDs
        recommended_movies = [self.user_movie_ratings.columns[idx] for idx in recommended_indices]
        
        return recommended_movies
    
    def item_based_recommendation(self, user_id, n=10):
        """
        Generate recommendations for a user based on item similarity
        
        Parameters:
        user_id (int): The user ID to generate recommendations for
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs recommended for the user
        """
        if self.item_similarity is None:
            self.compute_item_similarity()
        
        # Get the user's ratings
        user_idx = self.user_movie_ratings.index.get_loc(user_id)
        user_ratings = self.user_movie_ratings.iloc[user_idx].values
        
        # Find movies the user has already rated
        rated_indices = user_ratings.nonzero()[0]
        
        # Calculate predicted ratings for all movies
        predicted_ratings = np.zeros(len(user_ratings))
        
        for item_idx in range(len(user_ratings)):
            # Skip already rated movies
            if user_ratings[item_idx] > 0:
                predicted_ratings[item_idx] = -1
                continue
            
            # Calculate weighted sum of ratings for similar items
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_idx in rated_indices:
                # Get similarity between this item and the rated item
                sim = self.item_similarity[item_idx, rated_idx]
                
                # Skip if similarity is too low (optional)
                if sim <= 0:
                    continue
                
                # Add to weighted sum
                weighted_sum += sim * user_ratings[rated_idx]
                similarity_sum += abs(sim)
            
            # Calculate predicted rating if there are similar items
            if similarity_sum > 0:
                predicted_ratings[item_idx] = weighted_sum / similarity_sum
        
        # Get top N recommendations
        recommended_indices = predicted_ratings.argsort()[-n:][::-1]
        
        # Convert indices to movie IDs
        recommended_movies = [self.user_movie_ratings.columns[idx] for idx in recommended_indices]
        
        return recommended_movies


class ContentBasedFiltering:
    """
    Implements content-based filtering using movie genres and titles
    """
    def __init__(self, movies_df, genre_matrix):
        """
        Initialize with movies dataframe and genre matrix
        
        Parameters:
        movies_df (pandas.DataFrame): DataFrame containing movie information
        genre_matrix (pandas.DataFrame): Matrix of movies and their genres
        """
        self.movies_df = movies_df
        self.genre_matrix = genre_matrix
        self.title_vectorizer = None
        self.title_features = None
        self.genre_similarity = self._compute_genre_similarity()
        self._prepare_title_features()
    
    def _compute_genre_similarity(self):
        """
        Compute similarity between movies based on their genres
        """
        return cosine_similarity(self.genre_matrix)
    
    def _prepare_title_features(self):
        """
        Prepare TF-IDF features from movie titles
        """
        self.title_vectorizer = TfidfVectorizer(stop_words='english')
        self.title_features = self.title_vectorizer.fit_transform(
            self.movies_df['clean_title'].fillna('')
        )
    
    def recommend_by_genres(self, movie_id, n=10):
        """
        Recommend movies similar to the given movie based on genres
        
        Parameters:
        movie_id (int): The movie ID to find similar movies to
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs similar to the given movie
        """
        # Get the index of the movie in the genre matrix
        movie_idx = self.genre_matrix.index.get_loc(movie_id)
        
        # Get the similarity scores for this movie with all other movies
        movie_similarities = self.genre_similarity[movie_idx]
        
        # Sort the movies by similarity
        similar_indices = movie_similarities.argsort()[-n-1:][::-1]
        
        # Remove the movie itself
        similar_indices = similar_indices[similar_indices != movie_idx][:n]
        
        # Convert indices to movie IDs
        similar_movies = [self.genre_matrix.index[idx] for idx in similar_indices]
        
        return similar_movies
    
    def recommend_by_title(self, movie_id, n=10):
        """
        Recommend movies similar to the given movie based on title similarity
        
        Parameters:
        movie_id (int): The movie ID to find similar movies to
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs with similar titles to the given movie
        """
        # Get the index of the movie in the movies dataframe
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        
        # Compute cosine similarity between the movie title and all other movie titles
        title_similarity = cosine_similarity(
            self.title_features[movie_idx:movie_idx+1], 
            self.title_features
        ).flatten()
        
        # Sort the movies by similarity
        similar_indices = title_similarity.argsort()[-n-1:][::-1]
        
        # Remove the movie itself
        similar_indices = similar_indices[similar_indices != movie_idx][:n]
        
        # Convert indices to movie IDs
        similar_movies = [self.movies_df.iloc[idx]['movieId'] for idx in similar_indices]
        
        return similar_movies
    
    def recommend_by_content(self, movie_id, genre_weight=0.8, n=10):
        """
        Recommend movies similar to the given movie based on a weighted combination
        of genre and title similarity
        
        Parameters:
        movie_id (int): The movie ID to find similar movies to
        genre_weight (float): Weight to give to genre similarity (0-1)
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs similar to the given movie
        """
        # Get recommendations based on genres
        genre_recs = self.recommend_by_genres(movie_id, n=n*2)
        
        # Get recommendations based on title
        title_recs = self.recommend_by_title(movie_id, n=n*2)
        
        # Create a dictionary to store combined scores
        movie_scores = {}
        
        # Add genre-based recommendations with their weights
        for i, movie in enumerate(genre_recs):
            # Reverse the ranking to give higher score to higher ranked movies
            score = (len(genre_recs) - i) * genre_weight
            movie_scores[movie] = movie_scores.get(movie, 0) + score
        
        # Add title-based recommendations with their weights
        for i, movie in enumerate(title_recs):
            # Reverse the ranking to give higher score to higher ranked movies
            score = (len(title_recs) - i) * (1 - genre_weight)
            movie_scores[movie] = movie_scores.get(movie, 0) + score
        
        # Sort movies by their combined score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N movie IDs
        return [movie_id for movie_id, _ in sorted_movies[:n]]
    
    def recommend_by_genres_list(self, genres, n=10):
        """
        Recommend movies that match the given genres
        
        Parameters:
        genres (list): List of genres to match
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs that match the given genres
        """
        # Create a query vector with 1s for the specified genres
        query_vector = pd.Series(0, index=self.genre_matrix.columns)
        for genre in genres:
            if genre in query_vector.index:
                query_vector[genre] = 1
        
        # Compute similarity with all movies
        similarity = cosine_similarity(
            query_vector.values.reshape(1, -1),
            self.genre_matrix.values
        ).flatten()
        
        # Get the indices of the top N similar movies
        similar_indices = similarity.argsort()[-n:][::-1]
        
        # Convert indices to movie IDs
        similar_movies = [self.genre_matrix.index[idx] for idx in similar_indices]
        
        return similar_movies

class HybridRecommender:
    """
    Combines collaborative and content-based filtering for better recommendations
    """
    def __init__(self, collaborative_model, content_model, movies_df):
        """
        Initialize with collaborative and content-based models
        
        Parameters:
        collaborative_model (CollaborativeFiltering): Trained collaborative filtering model
        content_model (ContentBasedFiltering): Trained content-based filtering model
        movies_df (pandas.DataFrame): DataFrame containing movie information
        """
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.movies_df = movies_df
    
    def recommend_for_user(self, user_id, n=10, collaborative_weight=0.7):
        """
        Generate hybrid recommendations for a user
        
        Parameters:
        user_id (int): The user ID to generate recommendations for
        n (int): Number of recommendations to generate
        collaborative_weight (float): Weight to give to collaborative filtering (0-1)
        
        Returns:
        list: List of movie IDs recommended for the user
        """
        # Get collaborative filtering recommendations
        cf_recs = self.collaborative_model.user_based_recommendation(user_id, n=n*2)
        
        # For content-based recommendations, we need a movie the user has rated highly
        # Get the user's ratings
        user_idx = self.collaborative_model.user_movie_ratings.index.get_loc(user_id)
        user_ratings = self.collaborative_model.user_movie_ratings.iloc[user_idx]
        
        # Find the movies the user has rated highly (rating >= 4)
        highly_rated = user_ratings[user_ratings >= 4].index.tolist()
        
        # If the user hasn't rated any movies highly, use the movies they've rated
        if not highly_rated:
            highly_rated = user_ratings[user_ratings > 0].index.tolist()
        
        # If the user hasn't rated any movies, return collaborative filtering recommendations
        if not highly_rated:
            return cf_recs[:n]
        
        # Select a random highly rated movie for content-based recommendations
        import random
        random_movie_id = random.choice(highly_rated)
        
        # Get content-based recommendations
        cb_recs = self.content_model.recommend_by_content(random_movie_id, n=n*2)
        
        # Combine recommendations with weights
        movie_scores = {}
        
        # Add collaborative filtering recommendations with their weights
        for i, movie in enumerate(cf_recs):
            # Reverse the ranking to give higher score to higher ranked movies
            score = (len(cf_recs) - i) * collaborative_weight
            movie_scores[movie] = movie_scores.get(movie, 0) + score
        
        # Add content-based recommendations with their weights
        for i, movie in enumerate(cb_recs):
            # Reverse the ranking to give higher score to higher ranked movies
            score = (len(cb_recs) - i) * (1 - collaborative_weight)
            movie_scores[movie] = movie_scores.get(movie, 0) + score
        
        # Sort movies by their combined score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N movie IDs
        return [movie_id for movie_id, _ in sorted_movies[:n]]
    
    def recommend_by_genres(self, genres, n=10):
        """
        Recommend movies based on specified genres
        
        Parameters:
        genres (list): List of genres to match
        n (int): Number of recommendations to generate
        
        Returns:
        list: List of movie IDs that match the given genres
        """
        return self.content_model.recommend_by_genres_list(genres, n=n)
    
    def get_movie_details(self, movie_ids):
        """
        Get details for the recommended movies
        
        Parameters:
        movie_ids (list): List of movie IDs
        
        Returns:
        list: List of dictionaries containing movie details
        """
        movies = []
        for movie_id in movie_ids:
            movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            movies.append({
                'id': movie_id,
                'title': movie['title'],
                'genres': movie['genres'],
                'year': movie['year']
            })
        return movies