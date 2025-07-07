# Standard library imports
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import joblib
import pandas as pd
import torch

# Add the parent directory of 'notebooks' to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Local application imports
from config import (
    MODEL_FILEPATH,
    TARGET_COLUMN,
    TRAIN_SET_FILEPATH,
    SCALER_FILEPATH,
    RECOMMENDATION_OUTPUT_DIRECTORY,
)

from src.model import Net  # Your trained neural network model class

from src.acoustic_brainz_utils import (
    fetch_audio_features_from_acoustic_brainz,
)

from src.music_brainz_utils import (
    get_metadata_features_from_music_brainz,
    get_album_tracks,
    fetch_random_album_batch,
)

from src.transform_training_data import (
    transform_audio_features,
    RunMode,
)

from src.prediction_utils import (
    fetch_valid_albums_for_prediction,
    gather_features_for_prediction,
)

'''
    Load a trained PyTorch model and prepare it for inference.

    Parameters:
        input_dim (int): The number of input features expected by the model.

    Returns:
        torch.nn.Module: The loaded model set to evaluation mode.
'''
def load_default_trained_model(input_dim: int) -> torch.nn.Module:
    model = Net(input_dim)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()
    return model

'''
    Cleans and transforms a single row of raw features for model inference.

    This method:
    - Converts the raw feature dictionary into a DataFrame.
    - Applies a transformation function (e.g., scaling, encoding).
    - Fills in any missing expected columns with zeros.
    - Reorders the columns to match the expected model input schema.

    Args:
        row_features (dict): Dictionary of combined raw features for a single track.
        feature_columns (List[str]): List of columns (in order) expected by the model.
        scaler: A fitted scaler (e.g., MinMaxScaler) used to transform numeric inputs.

    Returns:
        pd.DataFrame: A single-row DataFrame with all required features in correct order.
                      Returns an empty DataFrame if transformation fails.
'''
def clean_and_transform_features(
    row_features: dict,
    feature_columns: List[str],
    scaler
) -> pd.DataFrame:
    # Convert raw dictionary to a single-row DataFrame
    raw_df = pd.DataFrame([row_features])

    # Apply custom transformation logic (e.g., encoding, scaling)
    transformed_df = transform_audio_features(
        df=raw_df,
        mode=RunMode.PREDICTION,
        scaler=scaler,
        expected_columns=feature_columns
    )

    # If transformation returned an empty DataFrame, exit early
    if transformed_df.empty:
        return pd.DataFrame()

    # Add any missing expected columns, initializing them to 0.0
    missing_cols = [col for col in feature_columns if col not in transformed_df.columns]
    for col in missing_cols:
        transformed_df[col] = 0.0

    # Reorder columns to match the model's expected input order
    transformed_df = transformed_df[feature_columns]

    return transformed_df

'''
    Runs model inference on a single track's feature DataFrame.

    Args:
        model (torch.nn.Module): The trained PyTorch binary classification model.
        features_df (pd.DataFrame): A single-row DataFrame of transformed features.

    Returns:
        float: The predicted probability score (between 0 and 1).
'''
def predict_track_score(model: torch.nn.Module, features_df: pd.DataFrame) -> float:
    x = torch.tensor(features_df.values, dtype=torch.float32)
    with torch.no_grad():
        return model(x).item()

'''
    Predicts how likely you are to like each track using a trained binary classifier.

    Parameters:
        model (torch.nn.Module): Trained PyTorch model for binary classification.
        mbid_title_list (List[Tuple[str, str]]): List of (MBID, track title) pairs.
        feature_columns (List[str]): Names of features required by the model in the correct order.
        scaler: The scaler used to scale features during training.

    Returns:
        Tuple[List[Tuple[str, float]], pd.DataFrame]:
            - A list of (track title, predicted probability), where -1.0 indicates missing data.
            - A DataFrame of all transformed feature rows used for predictions.
'''
def predict_likes(
    model: torch.nn.Module,
    mbid_title_list: List[Tuple[str, str]],
    feature_columns: List[str],
    scaler
) -> Tuple[List[Tuple[str, float]], pd.DataFrame]:
    
    results = []
    prediction_inputs = pd.DataFrame(columns=feature_columns).astype("float32")
    scaler = joblib.load(SCALER_FILEPATH)

    for mbid, title in mbid_title_list:
        features = gather_features_for_prediction(mbid)

        if features:
            row_features = features
            
            transformed_audio_features_df = clean_and_transform_features(
                row_features=row_features,
                feature_columns=feature_columns,
                scaler=scaler
            )

            # Only proceed if transformed_audio_features_df is not empty
            if not transformed_audio_features_df.empty:
                new_row_df = pd.DataFrame(transformed_audio_features_df.values, columns=feature_columns)
                if new_row_df.notna().any(axis=None):
                    prediction_inputs = pd.concat([prediction_inputs, new_row_df], ignore_index=True)

                score = predict_track_score(model, transformed_audio_features_df)
                results.append((title, score))
            else:
                results.append((title, -1.0))  # Feature gathering failed
        else:
            results.append((title, -1.0)) # If we can't get data for a track, indicate with -1.0

    return results, prediction_inputs

'''
    Save album recommendation results, including track predictions and feature inputs, to a CSV file.

    Args:
        album_name (str): The name of the recommended album.
        artist (str): The name of the artist.
        predictions (list[tuple[str, float]]): List of (track title, predicted score) tuples.
        prediction_inputs (pd.DataFrame): DataFrame of transformed features used for prediction.

    Returns:
        str: The full path to the saved CSV file.
'''
def save_album_recommendation_to_csv(
    album_name: str,
    artist: str,
    predictions: list[tuple[str, float]],
    prediction_inputs: pd.DataFrame
) -> str:
    # Create a DataFrame from the predictions list with specified columns
    recommendation_df = pd.DataFrame(predictions, columns=["Track Title", "Predicted Score"])

    # Insert artist and album columns at the beginning of the DataFrame
    recommendation_df.insert(0, "Artist", artist)
    recommendation_df.insert(0, "Album", album_name)

    # If prediction_inputs is not empty and has at least one non-NaN column, 
    # concatenate it to the right of the recommendation DataFrame
    if not prediction_inputs.empty and not prediction_inputs.dropna(axis=1, how='all').empty:
        recommendation_df = pd.concat(
            [recommendation_df.reset_index(drop=True), prediction_inputs.reset_index(drop=True)],
            axis=1
        )

    # Sanitize the album name to create a safe filename by removing problematic characters
    safe_album_name = re.sub(r'[\\/*?:"<>|]', "", album_name).strip().replace(" ", "_")
    filename = f"album_recommendation_{safe_album_name}.csv"

    # Construct the full path for saving the CSV
    full_path = os.path.join(RECOMMENDATION_OUTPUT_DIRECTORY, filename)

    # Save the DataFrame to a CSV file without the DataFrame index
    recommendation_df.to_csv(full_path, index=False)

    print(f"\n📁 Saved recommendation to {full_path}")

    return full_path

'''
    Prints album status messages consistently based on the context.

    Args:
        album_name (str): Name of the album.
        artist (str): Name of the artist.
        status (str): One of 'recommended', 'skipped', or 'not_recommended' indicating the album status.
        predictions (list[tuple[str, float]], optional): List of (track title, predicted score). Required for
                                                        'recommended' and 'not_recommended' statuses to print track details.
        liked_song_threshold (float, optional): Threshold to determine if a track is liked. Defaults to 0.5.
        liked_songs (int, optional): Number of liked tracks, used for 'not_recommended' status.
        total_tracks (int, optional): Total number of tracks, used for 'not_recommended' status.
        album_recommendation_threshold (float, optional): Threshold proportion for recommending album,
                                                          used for 'not_recommended' status.
'''
def print_album_status(
    album_name: str,
    artist: str,
    status: str,
    predictions: list[tuple[str, float]] = None,
    liked_song_threshold: float = 0.5,
    liked_songs: int = None,
    total_tracks: int = None,
    album_recommendation_threshold: float = None
) -> None:
    if status == 'recommended':
        print(f"\n💿 Album Recommendation Found!")
        print(f"Album: {album_name}")
        print(f"Artist: {artist}")
        print("\nTrack Predictions:")
        for title, score in predictions or []:
            emoji = '👍' if score >= liked_song_threshold else '👎'
            print(f" - {title}: {emoji} ({score:.2f})")

    elif status == 'skipped':
        print(f"\n❌ Skipping album due to insufficient data to predict likes.")
        print(f"Album: {album_name}")
        print(f"Artist: {artist}")

    elif status == 'not_recommended':
        print(f"\n❌ Not enough liked tracks to recommend this album.")
        print(f"Album: {album_name}")
        print(f"Artist: {artist}")
        if liked_songs is not None and total_tracks is not None and album_recommendation_threshold is not None:
            print(f"Liked {liked_songs} out of {total_tracks} tracks "
                  f"(Threshold: {album_recommendation_threshold * 100:.0f}%)")
        print("\nTrack Predictions:")
        for title, score in predictions or []:
            emoji = '👍' if score >= liked_song_threshold else '👎'
            print(f" - {title}: {emoji} ({score:.2f})")
    else:
        raise ValueError(f"Unknown status '{status}'. Expected 'recommended', 'skipped', or 'not_recommended'.")

'''
    Loads training feature columns (excluding label) and the trained model.

    Returns:
        Tuple[torch.nn.Module, List[str], Any]: 
            - model: Trained PyTorch model.
            - feature_columns: List of column names used as model inputs.
            - scaler: Fitted feature scaler used during training.
'''
def load_model_features_scaler() -> Tuple[torch.nn.Module, List[str], Any]:
    train_df = pd.read_csv(TRAIN_SET_FILEPATH)
    feature_columns = train_df.columns.drop('liked').tolist()
    model = load_default_trained_model(input_dim=len(feature_columns))
    scaler = joblib.load(SCALER_FILEPATH)
    return model, feature_columns, scaler
    
'''
    Continuously searches for an album you are likely to enjoy based on a trained model.

    This function loads a trained neural network song recommender and uses it to predict
    how much you will like tracks from randomly fetched albums. An album is recommended
    if the proportion of tracks predicted as "liked" exceeds a specified threshold. 
    The first suitable album's predictions are saved to a CSV file and printed to stdout.

    Args:
        liked_song_threshold (float, optional): Minimum prediction score (0–1) to consider a song "liked".
                                                Defaults to 0.5.
        album_recommendation_threshold (float, optional): Minimum proportion of "liked" songs on an album
                                                          to recommend the album. Defaults to 0.5.
        album_track_length_threshold (int, optional): Minimum number of tracks required on an album
                                                      for it to be considered for recommendation.
                                                      Defaults to 0 (no minimum).

    Returns:
        None

    Side Effects:
        - Prints album and track predictions to stdout.
        - Saves the first recommended album's predictions to a CSV file.
        - Repeats until a suitable album is found based on model predictions and filtering criteria.
'''
def search_for_album_you_like(
    liked_song_threshold: float = 0.5,
    album_recommendation_threshold: float = 0.5,
    album_track_length_threshold: int = 0
) -> None:
    model, feature_columns, scaler = load_model_features_scaler()

    # Continuously fetch random albums until a recommendation is found
    while True:
        print("Fetching initial albums...")
        for album, tracks in fetch_valid_albums_for_prediction(album_track_length_threshold):
            album_id = album['id']
            album_name = album.get('title', 'Unknown')
            artist = ", ".join(
                [a['artist']['name'] for a in album.get('artist-credit', []) if 'artist' in a]
            )

            # Predict liked tracks using the trained model
            predictions, prediction_inputs = predict_likes(
                model, tracks, feature_columns, scaler
            )
            if not predictions:
                continue
    
            # Count liked and skipped songs based on prediction scores
            liked_songs = sum(1 for _, score in predictions if score >= liked_song_threshold)
            skipped_songs = sum(1 for _, score in predictions if score == -1.0)
    
            # Decide whether to recommend the album based on thresholds
            if liked_songs / len(predictions) >= album_recommendation_threshold:
                # Recommended album
                print_album_status(
                    album_name=album_name,
                    artist=artist,
                    status='recommended',
                    predictions=predictions,
                    liked_song_threshold=liked_song_threshold
                )
    
                # Save recommendation details to CSV file
                save_album_recommendation_to_csv(
                    album_name=album_name,
                    artist=artist,
                    predictions=predictions,
                    prediction_inputs=prediction_inputs
                )
                return  # Stop searching after first recommendation
    
            elif skipped_songs / len(predictions) > 0.5:
                # Skipped album due to lack of data
                print_album_status(
                    album_name=album_name,
                    artist=artist,
                    status='skipped'
                )
    
            else:
                # Not enough liked tracks
                print_album_status(
                    album_name=album_name,
                    artist=artist,
                    status='not_recommended',
                    predictions=predictions,
                    liked_song_threshold=liked_song_threshold,
                    liked_songs=liked_songs,
                    total_tracks=len(predictions),
                    album_recommendation_threshold=album_recommendation_threshold
                )

            print("\nFetching more albums...")