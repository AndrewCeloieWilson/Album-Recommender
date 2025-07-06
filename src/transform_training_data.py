import os
import re
import sys
from enum import Enum
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

# Add parent directory to sys.path for module resolution
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import SCALER_FILEPATH

class RunMode(Enum):
    TRAINING = "training"
    PREDICTION = "prediction"

# Define a mapping from raw tokens to canonical genre labels
GENRE_MAP = {
    "rock": "rock",
    "pop": "pop",
    "hip-hop": "hip hop",
    "hip hop": "hip hop",
    "rap": "rap",
    "r&b": "r&b",
    "rhythm and blues": "r&b",
    "rhythm & blues": "r&b",
    "soul": "soul",
    "jazz": "jazz",
    "country": "country",
    "electronic": "electronic",
    "edm": "electronic",
    "metal": "metal",
    "thrash metal": "metal",
    "alternative metal": "metal",
    "industrial metal": "metal",
    "folk": "folk",
    "americana": "folk",
    "blues": "blues",
    "indie": "indie",
    "indie rock": "indie",
    "indie pop": "indie",
    "dance": "dance",
    "funk": "funk",
    "punk": "punk",
    "alternative": "alternative",
    "classical": "classical",
    "acoustic": "acoustic",
    "lo-fi": "lo-fi",
    "singer-songwriter": "singer-songwriter",
    "singer/songwriter": "singer-songwriter",
    "soundtrack": "soundtrack",
    "industrial": "industrial",
    "house": "house",
    "techno": "techno",
    "ambient": "ambient",
    "unknown": "unknown",
    "null": "unknown",
    "none": "unknown",
    "": "unknown"
}

# Compile regex for token splitting (delimiters: comma, semicolon, pipe, slash, arrow, dash)
SPLIT_REGEX = re.compile(r"[,\|;/>\-]+")


'''
    Transforms and prepares a song feature DataFrame for neural network input.

    This function modularly cleans and preprocesses acoustic and metadata features for training or prediction.
    It performs column dropping, missing value handling, numeric scaling, categorical encoding, 
    float32 conversion, and column alignment for inference.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw song features.
        mode (RunMode): The current mode, either TRAINING or PREDICTION.
        scaler (MinMaxScaler, optional): A pre-fitted scaler used only in prediction mode.
        expected_columns (list[str], optional): A list of feature columns required by the model (for column alignment during prediction).

    Returns:
        pd.DataFrame: The cleaned, encoded, and scaled feature matrix ready for neural network input.
'''
def transform_audio_features(
    df: pd.DataFrame, 
    mode: RunMode, 
    scaler: Optional[MinMaxScaler] = None, 
    expected_columns: Optional[list[str]] = None
) -> pd.DataFrame:
                
    df = drop_unused_columns(df)
    df = drop_critical_na_rows(df, mode)        
    df = scale_numeric_features(df, mode, scaler)
    df = encode_categorical_features(df)
    if mode == RunMode.PREDICTION:
        df = add_missing_prediction_columns(df, expected_columns)
    df = df.round(4)
    df = df.astype(np.float32)
    
    return df

'''
    Ensures the DataFrame has all expected feature columns required by the model during prediction.

    This is especially useful for re-adding one-hot encoded columns that were not present 
    in the input data (e.g., a genre or key that didn’t appear in a given song).

    Args:
        df (pd.DataFrame): The prediction input DataFrame with encoded features.
        expected_columns (list[str]): Full list of columns the model was trained on.

    Returns:
        pd.DataFrame: A DataFrame padded with any missing columns (filled with 0.0).
'''
def add_missing_prediction_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        missing_df = pd.DataFrame({col: [0.0] for col in missing_cols})
        df = pd.concat([df, missing_df], axis=1)
    
    return df

'''
    Encodes categorical musical features including scale type, key, chords, and genre.

    - Converts `key_scale` and `chords_scale` from 'major'/'minor' to binary (1/0).
    - One-hot encodes `key_key` and `chords_key` columns with appropriate prefixes.
    - Normalizes raw `genre` strings into canonical labels and one-hot encodes them.
    - Drops original raw categorical columns after encoding.

    Args:
        df (pd.DataFrame): The input DataFrame with raw categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features ready for modeling.
'''
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Binary encode scale types
    df['key_scale'] = df['key_scale'].map({'major': 1, 'minor': 0})
    df['chords_scale'] = df['chords_scale'].map({'major': 1, 'minor': 0})

    # One-hot encode musical key and chord root
    df = pd.get_dummies(df, columns=['key_key', 'chords_key'], prefix=['key', 'chords'])

    # Optional
    # Add or remove this block to include or remove genre data from the recommender system
    
    # Normalize and one-hot encode genres
    df["normalized_genres"] = df["genre"].apply(normalize_genre)
    filtered_genres = df["normalized_genres"].apply(lambda genres: [g for g in genres if g != "unknown"])
    mlb = MultiLabelBinarizer()
    one_hot_normalized_genres = pd.DataFrame(
       mlb.fit_transform(filtered_genres),
       columns=mlb.classes_,
       index=df.index
    )
    one_hot_normalized_genres = one_hot_normalized_genres.add_prefix("genre_")
    df = df.join(one_hot_normalized_genres)
    
    df.drop(columns=["genre", "normalized_genres"], inplace=True)

    # End Optional

    return df

'''
    Scales numeric features using MinMaxScaler for model compatibility.

    In training mode:
    - Identifies numeric columns (excluding the 'liked' label).
    - Fits a new MinMaxScaler on those columns.
    - Optionally saves the scaler using joblib.

    In prediction mode:
    - Uses the provided scaler to transform the same set of numeric columns.

    Args:
        df (pd.DataFrame): The DataFrame to be scaled.
        mode (RunMode): Whether the operation is for TRAINING or PREDICTION.
        scaler (MinMaxScaler, optional): A pre-fitted scaler for prediction mode.

    Returns:
        pd.DataFrame: The scaled DataFrame
'''
def scale_numeric_features(df: pd.DataFrame, mode: RunMode, scaler: MinMaxScaler = None) -> pd.DataFrame:
    # Identify numeric columns to scale (excluding 'liked')
    columns_to_scale = df.select_dtypes(include=['float32', 'float64', 'int']).columns.difference(['liked'])

    if mode == RunMode.PREDICTION:
        # Use pre-fitted scaler to transform numeric columns
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    else:
        # Fit and apply a new scaler during training
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        joblib.dump(scaler, SCALER_FILEPATH)

    return df

'''
    Drops rows with missing values in critical categorical columns during training.

    Only applies this operation if mode is RunMode.TRAINING. This ensures that
    key musical attributes required for model learning are always present.

    Args:
        df (pd.DataFrame): The input DataFrame.
        mode (RunMode): Indicates whether the function is being called during training or prediction.

    Returns:
        pd.DataFrame: DataFrame with rows containing missing critical values dropped (if training).
'''
def drop_critical_na_rows(df: pd.DataFrame, mode: RunMode) -> pd.DataFrame:
    if mode == RunMode.TRAINING:
        critical_columns = ['key_key', 'key_scale', 'chords_key', 'chords_scale', 'release_year']
        # Optional 
        # Add or remove release_year to the above list if you are using release_year in your data and want to drop any training data that doesn't include it
        df = df.dropna(subset=critical_columns)
    return df

'''
    Removes non-modeling metadata columns.

    This helper method performs:
    - Removal of ID, title, and label metadata not used by the model.

    Args:
        df (pd.DataFrame): Input DataFrame containing song metadata and audio features.

    Returns:
        pd.DataFrame: Cleaned DataFrame with unnecessary columns and/or incomplete rows removed.
'''
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop metadata columns not used for model training or prediction
    drop_columns = ['mbid', 'artist', 'label', 'release_title', 'recording_title']
    # Optional 
    # Add or remove release_year to the above list depending on if you are using release_year in your data or not and if you want to 
    # drop the column
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Optional 
    # Add this line if you want to remove all genre data. Comment this line if you want to include it.
    #df = df.drop(columns=[col for col in df.columns if col.startswith("genre")])
    # End Optional
    
    return df

'''
    Normalizes a raw genre string into a list of cleaned, canonical genres.
    
    Args:
        raw_genre (str): The raw genre metadata string from the dataset.
        
    Returns:
        List[str]: A list of normalized genre labels (e.g., ["rock", "pop"]).
'''
def normalize_genre(raw_genre: str) -> list[str]:
    # Return ["unknown"] if input is missing, null, or non-string
    if not isinstance(raw_genre, str) or raw_genre.strip().lower() in {"", "null", "unknown", "none"}:
        return ["unknown"]
    
    # Use regex to split genre string on common delimiters: comma, pipe, semicolon, slash, greater-than, dash
    tokens = re.split(r"[,\|;/>\-]+", raw_genre.lower())

    # Clean and map each token to its canonical form
    normalized_genres = set()
    for token in tokens:
        clean_token = token.strip().lower()
        if clean_token == "":
            continue
        mapped_genre = GENRE_MAP.get(clean_token, "unknown")
        normalized_genres.add(mapped_genre)
    
    return list(normalized_genres)