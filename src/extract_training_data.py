import os
import sys
import time
from typing import Dict, Optional

import pandas as pd

# Add parent directory to sys.path for module resolution
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    LIKED_DISLIKED_FILEPATH,
    LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH,
)

from src.acoustic_brainz_utils import (
    get_high_level_audio_features_from_acoustic_brainz,
    get_low_level_audio_features_from_acoustic_brainz,
    fetch_audio_features_from_acoustic_brainz,
)

from src.music_brainz_utils import get_metadata_features_from_music_brainz

'''
    Loads track data, enriches it with audio features from AcousticBrainz, and saves the augmented dataset.

    This function reads a CSV file containing MusicBrainz recording IDs ('mbid') and 'liked' labels,
    retrieves detailed audio features for each track by querying the AcousticBrainz API,
    and saves the enriched dataset to a specified CSV file for downstream modeling or analysis.

    Args:
        None

    Returns:
        pd.DataFrame: DataFrame containing audio features combined with 'mbid' and 'liked' labels.
'''
def extract_and_save_raw_audio_features() -> pd.DataFrame:
    # Load the input CSV of liked and disliked tracks
    liked_and_disliked_tracks_df = pd.read_csv(LIKED_DISLIKED_FILEPATH)
    print(f"📥 Loaded {len(liked_and_disliked_tracks_df)} tracks from '{LIKED_DISLIKED_FILEPATH}'.")

    # Fetch audio features for each MBID in the DataFrame
    features_df = collect_audio_features(liked_and_disliked_tracks_df)

    # Save the resulting DataFrame to CSV
    features_df.to_csv(LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH, index=False)
    print(f"✅ Saved audio features to '{LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH}'")

    return features_df

'''
    Collects and aggregates audio features for tracks identified by MusicBrainz IDs.

    Iterates through each row in the input DataFrame, retrieves audio features from the AcousticBrainz API 
    and metadata from the MusicBrainz API for each MusicBrainz recording ID (MBID). Combines these features 
    with the original 'liked' label and MBID. Includes delays between API calls to respect rate limits.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the following columns:
            - 'mbid' (str): MusicBrainz recording IDs.
            - 'liked' (int): Binary label indicating if the track is liked (1) or not (0).

    Returns:
        pd.DataFrame: A DataFrame where each row contains aggregated audio features and metadata for a track,
                      including the original 'liked' label and 'mbid'.
'''
def collect_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    features_list = []  # Will hold dictionaries of track metadata + AB features

    for idx, row in df.iterrows():
        mbid = row["mbid"]
        if mbid:
            row_features = {}
            time.sleep(1.0)  # Respect rate limits
            acoustic_brainz_features = fetch_audio_features_from_acoustic_brainz(mbid)         
            if acoustic_brainz_features:
                row_features.update(acoustic_brainz_features)
            
                time.sleep(1.0) # Respect rate limits
                music_brainz_features = get_metadata_features_from_music_brainz(mbid)
                row_features.update(music_brainz_features)
            
                row_features["liked"] = row["liked"]
                row_features["mbid"] = mbid
                features_list.append(row_features)
            print(f"[{idx+1}/{len(df)}] Processed MBID {mbid} -> {'✅' if mbid and acoustic_brainz_features else '❌'}")

    return pd.DataFrame(features_list)
