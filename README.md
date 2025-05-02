# Movie Recommendation System

A sophisticated movie recommendation system built with Python that uses both collaborative filtering and content-based approaches to suggest movies to users based on their preferences or selected genres.

## Features

- **User-Based Collaborative Filtering**: Recommends movies based on what similar users liked
- **Item-Based Collaborative Filtering**: Recommends movies similar to those the user has liked
- **Content-Based Filtering**: Recommends movies based on genres and movie titles
- **Hybrid Recommendations**: Combines collaborative and content-based approaches for better recommendations
- **Genre-Based Recommendations**: Suggests movies that match selected genres
- **Search Functionality**: Allows users to search for movies by title
- **Rating System**: Users can rate movies to improve future recommendations
- **Web Interface**: Easy-to-use Flask web application

## Project Structure

```
movie-recommendation-system/
├── app.py                 
├── data_preparation.py     
├── recommendation_models.p
├── run_movie_recommender.py
├── README.md               
├── templates/              
│   └── index.html          
├── data/                   
└── models/                 
```

## Requirements

- Python 3.6 or higher
- Libraries: pandas, numpy, scikit-learn, Flask, requests

## Installation and Setup

1. Clone this repository or download all the files
2. Make sure you have Python 3.6+ installed
3. The system will automatically install required packages on first run

## Running the System

### Option 1: Using the run script (Recommended)

Simply run:

```bash
python run_movie_recommender.py
```

This will:
- Check your Python version
- Install required packages if needed
- Download the MovieLens dataset (~1MB)
- Train the recommendation models
- Start the Flask web server
- Open the web interface in your default browser

### Option 2: Manual setup

1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn Flask requests
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open a web browser and go to: http://127.0.0.1:5000/

## How to Use the Web Interface

### User-Based Recommendations

1. Select a user ID from the dropdown
2. Choose the number of recommendations you want
3. Click "Get Recommendations"
4. Browse the recommended movies

### Genre-Based Recommendations

1. Select one or more genres that interest you
2. Choose the number of recommendations you want
3. Click "Get Recommendations"
4. Browse the recommended movies

### Search Movies

1. Type a movie title (or part of it) in the search box
2. Browse the search results
3. If you're logged in as a user, you can rate the movies

### Rating Movies

1. Click "Rate this movie" on any movie card
2. Select a rating from 1 to 5 stars
3. Click "Submit Rating"
4. Your recommendations will be automatically updated

## How It Works

### Data Preparation

The system uses the MovieLens dataset, which contains movie ratings from users. On first run, it:

1. Downloads the latest small MovieLens dataset
2. Processes the data to extract useful features (genres, years, etc.)
3. Creates necessary matrices for recommendation algorithms

### Recommendation Algorithms

#### Collaborative Filtering

- **User-Based**: Finds users similar to you and recommends movies they liked
- **Item-Based**: Finds movies similar to those you've liked

#### Content-Based Filtering

- **Genre-Based**: Recommends movies with similar genres to those you've liked
- **Title-Based**: Uses natural language processing to find movies with similar titles

#### Hybrid Approach

- Combines the strengths of both collaborative and content-based filtering
- Weights each approach based on the situation for better recommendations

## Extending the System

### Adding More Features

- Implement more advanced algorithms like matrix factorization
- Add movie posters and descriptions from TMDB API
- Create user accounts and authentication
- Add more filters like year range, movie length, etc.

### Using Your Own Dataset

To use your own dataset, modify the `data_preparation.py` file to load and process your data in the same format as the MovieLens dataset.

## Troubleshooting

### Common Issues

- **ModuleNotFoundError**: Make sure you've installed all required packages
- **Port already in use**: If Flask can't start because port 5000 is in use, change the port in `app.py`
- **Slow recommendations**: The first recommendation might be slow as models are trained on first use

### Getting Help

If you encounter any issues, please check the error messages for clues about what went wrong.

## License

This project uses the MovieLens dataset, which is distributed under the [Creative Commons License](https://grouplens.org/datasets/movielens/).
Enjoy discovering new movies to watch!