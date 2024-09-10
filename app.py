from flask import Flask, request, jsonify
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load the movie data
movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1, 2], names=['item_id', 'title', 'genres'])

# Define the hybrid recommendation function (use from hybrid_model.py)
from hybrid_model import get_movie_recommendations

@app.route('/')
def index():
    return "Welcome to the Movie Recommendation API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get data from request
    data = request.get_json()
    movie_title = data.get('title', '')

    # Get recommendations
    try:
        recommendations = get_movie_recommendations(movie_title)
        return jsonify({'recommendations': recommendations.tolist()})
    except IndexError:
        return jsonify({'error': 'Movie not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
