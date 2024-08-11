from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

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
    print(f"Mean Squared Error: {mse}")

    return model

# Train the model
model = train_model()

# Function to select songs based on weather condition and popularity
def select_songs_based_on_weather(weather_condition):
    # Encode the weather condition
    weather_encoded = label_encoder.transform([weather_condition])[0]

    # Filter the dataset for the given weather condition
    songs_for_weather = dataset[dataset['Weather'] == weather_encoded]

    # If there are no songs for the weather condition, return empty list
    if songs_for_weather.empty:
        return []

    # Use the model to predict the popularity for the given weather condition
    popularity_prediction = model.predict([[weather_encoded]])

    # Filter songs based on predicted popularity
    songs_for_weather_filtered = songs_for_weather[songs_for_weather['Popularity'] >= popularity_prediction[0]]

    # If there are more than 100 songs, select the top 100 songs based on popularity
    if len(songs_for_weather_filtered) > 100:
        songs_for_weather_filtered = songs_for_weather_filtered.nlargest(100, 'Popularity')

    # Return a random selection of up to 5 songs from the filtered list
    return random.sample(songs_for_weather_filtered['Track Name'].tolist(), min(5, len(songs_for_weather_filtered)))

@app.route('/')
def home():
    return "<h1>Welcome to </h1>"

@app.route('/get_songs', methods=['POST'])
def get_songs():
    data = request.get_json()
    weather_condition = data['weather_condition']

    # Get the list of songs for the given weather condition and popularity
    songs_for_weather = select_songs_based_on_weather(weather_condition)

    return jsonify({'songs': songs_for_weather})

if __name__ == '__main__':
    app.run(debug=True)
