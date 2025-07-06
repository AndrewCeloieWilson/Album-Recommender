import os

import re

# Directories
DATA_DIR = os.path.join('..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# API
HEADERS = {"User-Agent": "AlbumRecommender/1.0 (Your-Email-Address)"}
ACOUSTIC_BRAINZ_BASE_URL = "https://acousticbrainz.org"
MUSIC_BRAINZ_BASE_URL = "https://musicbrainz.org"

# Raw Data
LIKED_DISLIKED_FILEPATH = os.path.join(RAW_DIR, 'my_liked_and_disliked_tracks.csv')
LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH = os.path.join(RAW_DIR, 'liked_and_disliked_tracks_with_acoustic_and_music_brainz_features.csv')
TARGET_COLUMN = 'liked'

# Output Data
MODEL_FILEPATH = os.path.join(OUTPUT_DIR, 'models', 'song_pref_model_weights.pth')
SCALER_FILEPATH = os.path.join(OUTPUT_DIR, 'scaler', 'scaler.joblib')
TRAIN_SET_FILEPATH = os.path.join(OUTPUT_DIR, 'training_data', 'train_set.csv')
TEST_SET_FILEPATH = os.path.join(OUTPUT_DIR, 'training_data', 'test_set.csv')
WEIGHTS_TARGET_DIRECTORY = os.path.join(OUTPUT_DIR, 'weights')
RECOMMENDATION_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIR, 'recommendations')