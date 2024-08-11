from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = pd.read_csv('mainDataset.csv', encoding='latin1')  # Update with the actual path to your new dataset

# Encode the 'Weather' column
label_encoder = LabelEncoder()
dataset.loc[:, 'Weather'] = label_encoder.fit_transform(dataset['Weather'])


# Train the machine learning model
def train_model():
    X = dataset[['Weather']]  # Input feature
    y = dataset['Popularity']  # Output feature

    # Handle missing values in y
    imputer = SimpleImputer(strategy='mean')
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_imputed, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Mean Squared Error: {mse}")

    return model


# Train the model
model = train_model()


# Function to select songs based on weather condition and popularity
def select_songs_based_on_weather(weather_condition):
    try:
        # Encode the weather condition
        weather_encoded = label_encoder.transform([weather_condition])[0]
    except ValueError:
        # Handle case where the weather condition is not recognized
        logger.error(f"Weather condition '{weather_condition}' not recognized.")
        return []

    # Filter the dataset for the given weather condition
    songs_for_weather = dataset[dataset['Weather'] == weather_encoded]

    if songs_for_weather.empty:
        return []

    # Use the model to predict the popularity for the given weather condition
    popularity_prediction = model.predict([[weather_encoded]])
    logger.info(f"Predicted popularity for {weather_condition}: {popularity_prediction[0]}")

    # Filter songs based on predicted popularity
    songs_for_weather_filtered = songs_for_weather[songs_for_weather['Popularity'] >= popularity_prediction[0]]

    if songs_for_weather_filtered.empty:
        return ["No songs available for this weather condition."]

    if len(songs_for_weather_filtered) > 100:
        songs_for_weather_filtered = songs_for_weather_filtered.nlargest(100, 'Popularity')

    return random.sample(songs_for_weather_filtered['Track Name'].tolist(), min(5, len(songs_for_weather_filtered)))


@app.route('/')
def home():
    return "<h1>Welcome to the Weather-Based Music Recommendation API</h1>"


@app.route('/get_songs', methods=['POST'])
def get_songs():
    data = request.get_json()
    weather_condition = data['weather_condition']

    # Get the list of songs for the given weather condition and popularity
    songs_for_weather = select_songs_based_on_weather(weather_condition)

    return jsonify({'songs': songs_for_weather})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  # Set debug=False in production
