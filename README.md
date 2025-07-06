
# Album Recommender Project

## Overview

This project is a machine learning system designed to recommend music albums based on user preferences. It leverages raw audio features extracted from your liked and disliked tracks and trains a binary classification neural network model to predict your music preferences. The model is then used to recommend albums that align with your taste.

Using the included dataset of 1500 songs split fairly evenly (52%) between liked and disliked, I have been able to achieve an accuracy of about 85% prediction success on the train and test sets before the model begins to overfit. 

## Features

=== STEP 1: Data Extraction ===
-This function reads your 'my_liked_and_disliked_tracks.csv' from the raw data folder and gathers associated audio features from Acoustic Brainz and Music Brainz. 

-A set of 1500 liked and disliked songs has already been included, as well as the raw data this method extracts. This can be used without modification. If you want to train the neural network using that data, you can skip directly to the training step.

If you want to use your own data, you need to update the 'my_liked_and_disliked_tracks.csv' file in data/raw.
Your 'my_liked_and_disliked_tracks.csv' file needs to contain a column of 'mbid' IDs representing the unique music brainz ID of songs you like. A second column 'liked' should include a 1 or 0 representing if you like or dislike the given song. Ideally, include about 50% liked and 50% disliked songs.

For each track, this function fetches detailed audio features from AcousticBrainz and metadata from MusicBrainz
and saves an enriched CSV file including all features. This CSV will be saved in data/raw and will be the base dataset for training.

=== STEP 2: Model Training ===
Train the model on the extracted and transformed dataset.
This function performs:
- Loading and cleaning the raw feature CSV
- Scaling numeric features with MinMaxScaler and saving the scaler object to scaler.joblib
- Encoding categorical variables (like musical key and scale)
- Splitting into train and test sets
- Initializing the neural network architecture (from src.model.Net in src/model.py)
- Training the model using binary cross-entropy loss and Adam optimizer
- Periodic evaluation on train and test sets, and printing progress of average loss.
- The average loss represents the target liked/disliked value (1 or 0) minus the predicted value averaged over the epoch. 
- Early stopping if test accuracy threshold is hit
- Saves the trained model to song_pref_model_weights.pth in the output/models folder
- These hyperparameters control your neural network training process. Feel free to tune them to improve model performance or training speed.
- TEST_SIZE = Percentage of data reserved for testing the model
- NUMBER_OF_EPOCHS = Number of full passes over the training dataset
- BATCH_SIZE = Number of samples per training update (mini-batch)
- LEARNING_RATE = Step size for optimizer updates, smaller = slower but more precise
- ADAMS_OPTIMIZER_WEIGHT_DECAY = Regularization strength to avoid overfitting
- EVALUATE_EVERY_X_EPOCHS = How often to evaluate and print training progress, as well as check for the early stopping threshold set by TARGET_TEST_ACCURACY.
- TARGET_TEST_ACCURACY = Early stopping threshold: If test accuracy reaches this percentage, stop training and save the model.

=== STEP 3: Output Model Data ===
This function generates output based on the trained model and saves it in the data/output/weights folder
Produces csv files that give information on bias, grouped importance of features, top 10 features, and weights per layer of the neural network These can be used to glean information about what the neural network learned from the dataset, how it is making recommendations, and why it has overfit or underfit the dataset.

=== STEP 4: Album Recommendation Search ===
This function searches through MusicBrainz for albums matching your criteria:

- It scores each song using the trained model's predictions
- It determines if the album has enough total songs and enough liked songs to recommend
- Outputs or prints albums you are likely to enjoy based on learned preferences
- Please note that this is slow due to Music Brainz and Acoustic Brainz being Free APIs which allow one query per second, and due to a high number of Music Brainz albums not having the necessary low level features that the neural network has been trained on.

IMPORTANT: Before proceeding please update the HEADERS value in config.py to include your e-mail address

These thresholds guide your album recommendation logic:
- LIKED_SONG_THRESHOLD = Minimum predicted probability for a song to be considered "liked"
- ALBUM_RECOMMENDATION_THRESHOLD = Minimum percentage of "liked" album tracks needed to recommend the album
- ALBUM_TRACK_LENGTH_THRESHOLD = Minimum number of tracks in an album for it to qualify for recommendation

## Data Sources

This project utilizes data from the following external sources:

- **MusicBrainz:** A community-maintained open music encyclopedia that collects music metadata.  
  Website: https://musicbrainz.org  
  Documentation: https://musicbrainz.org/doc/About
  Data: https://musicbrainz.org/doc/MusicBrainz_Database

- **AcousticBrainz:** Provides audio features extracted from digital music tracks using machine learning.  
  Website: https://acousticbrainz.org
  Data: https://acousticbrainz.org/data

## Important Notes on Data Limitations

- AcousticBrainz is a free API service that stopped being updated in 2022. This means the dataset might be outdated for newer music.
- The API enforces a strict rate limit of **1 request per second**, which can slow down data collection.
- There is no straightforward way to filter for tracks that include low-level audio feature data, which can limit dataset completeness and model performance.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Configuration

Update the `config.py` file to include your email address in the `HEADERS` dictionary for polite and compliant API usage. For example:

```python
HEADERS = {"User-Agent": "AlbumRecommender/1.0 (your_email@example.com)"}
```

## Running the Project

The main execution flow is controlled by the notebook `main.ipynb`:

1. **Extract Raw Audio Features:** Pulls features from your liked and disliked music list.
2. **Train Model:** Cleans and preprocesses the data, then trains a neural network classifier.
3. **Output Model Data:** Outputs evaluation metrics and saves the model.
4. **Search for Album Recommendations:** Searches for albums that match your music preferences.

You can adjust parameters such as test size, number of epochs, batch size, learning rate, and early stopping criteria in the notebook.

## Project Structure

- `data/raw/`: Contains raw input CSV files of user liked/disliked song data and raw audio features extracted from Acoustic Brainz and Music Brainz.
- `data/output/`: Stores output CSVs including saved models, scalers, training data, weights, and recommendations.
- `src/`: Contains source code modules for feature extraction, data transformation, model training, prediction, and output.
- `config.py`: Configuration settings including file paths and API headers.

## Dependencies

This project depends on:

- pandas
- numpy
- scikit-learn
- torch (PyTorch)
- joblib

See `requirements.txt` for full version details.

## Contact

For questions or issues, please contact: Andrew.Celoie.Wilson@gmail.com
