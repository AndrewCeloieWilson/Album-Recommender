import random
import time
from typing import Dict, Optional

import requests

from config import (
    ACOUSTIC_BRAINZ_BASE_URL,
    HEADERS,
)

'''
    Fetches selected high-level audio features and metadata from the AcousticBrainz API
    for a given MusicBrainz recording MBID.

    This function sends a request to the AcousticBrainz high-level endpoint and extracts
    a comprehensive set of audio feature probabilities and track metadata that can be
    used for audio classification, recommendation, or analysis.

    Args:
        mbid (str): The MusicBrainz recording ID of the track to retrieve features for.

    Returns:
        Optional[dict]: A dictionary containing extracted audio features and metadata,
            including but not limited to:
            - danceable (float): Probability that the track is danceable.
            - mood_* (float): Probabilities for moods such as acoustic, aggressive, happy, etc.
            - genre_* (float): Probabilities for subgenres from multiple classifiers.
            - ismir04_rhythm_* (float): Probabilities for rhythm categories.
            - moods_mirex_* (float): Probabilities for MIREX mood clusters.
            - timbre, tonal_atonal, voice_instrumental (float): Various audio characteristics probabilities.
            - danceability_confidence, voice_instrumental_confidence, timbre_confidence, tonal_atonal_confidence (float): Confidence scores.
            - instrumentalness_confidence (float): Derived confidence score for instrumentalness.
            - artist, genre, label (str): Track metadata tags joined as strings.

        Returns None if the MBID is invalid, the API request fails, or required data is missing.
'''
def get_high_level_audio_features_from_acoustic_brainz(mbid: str) -> Optional[dict]:
    try:
        response = requests.get(f"{ACOUSTIC_BRAINZ_BASE_URL}/api/v1/{mbid}/high-level", headers= HEADERS)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        features = {}
        highlevel = data.get("highlevel", {})
        metadata_tags = data.get("metadata", {}).get("tags", {})
        
        # Danceability score
        features["danceable"] = highlevel.get("danceability", {}).get("all", {}).get("danceable")
    
        # Genre subgenre probabilities
        for genre_key in ["genre_dortmund", "genre_electronic", "genre_rosamerica", "genre_tzanetakis"]:
            genre_data = highlevel.get(genre_key, {}).get("all", {})
            for subgenre, score in genre_data.items():
                features[f"{genre_key}_{subgenre}"] = score
    
        # ISMIR rhythm category probabilities
        rhythm_data = highlevel.get("ismir04_rhythm", {}).get("all", {})
        for rhythm, score in rhythm_data.items():
            features[f"ismir04_rhythm_{rhythm}"] = score
    
        # Mood probabilities
        for mood_key in [
            "mood_acoustic", "mood_aggressive", "mood_electronic",
            "mood_happy", "mood_party", "mood_relaxed", "mood_sad"
        ]:
            features[mood_key] = highlevel.get(mood_key, {}).get("probability")
    
        # Moods_mirex clusters
        mirex_data = highlevel.get("moods_mirex", {}).get("all", {})
        for cluster, score in mirex_data.items():
            features[f"moods_mirex_{cluster}"] = score
    
        # Timbre, Tonal/Atonal, Voice/Instrumental probabilities
        for key in ["timbre", "tonal_atonal", "voice_instrumental"]:
            features[key] = highlevel.get(key, {}).get("probability")

        # Add confidence values for select high-level features
        for key in ["danceability", "voice_instrumental", "timbre", "tonal_atonal"]:
            confidence = highlevel.get(key, {}).get("all", {}).get("probability")
            if confidence is not None:
                features[f"{key}_confidence"] = confidence

        # Instrumentalness confidence (from voice_instrumental, inverted logic)
        if "voice_instrumental_confidence" in features:
            features["instrumentalness_confidence"] = 1.0 - features["voice_instrumental_confidence"]

        # Basic metadata from tags
        features["artist"] = join_tags(metadata_tags.get("artist", []))
        features["genre"] = join_tags(metadata_tags.get("genre", []))
        features["label"] = join_tags(metadata_tags.get("label", []))
        
        return features
        
    except Exception as e:
        print(f"Error while fetching high-level features for MBID {mbid}: {e}")
        return None

'''
    Fetches selected low-level audio features and metadata from AcousticBrainz for a given MusicBrainz recording ID (MBID).

    This function queries the AcousticBrainz low-level endpoint and extracts a set of tonal, rhythmic, spectral,
    dynamic, and MFCC features that describe the audio characteristics of a track.

    Args:
        mbid (str): The MusicBrainz recording ID of the track.

    Returns:
        Optional[Dict[str, object]]: A dictionary containing extracted low-level audio features including:
            - key_key (str or None)
            - key_scale (str or None)
            - chords_key (str or None)
            - chords_scale (str or None)
            - chords_changes_rate (float or None)
            - chords_number_rate (float or None)
            - bpm (float or None)
            - beats_count (int or None)
            - average_loudness (float or None)
            - dynamic_complexity (float or None)
            - duration (float or None)
            - spectral_centroid (float or None)
            - spectral_rolloff (float or None)
            - spectral_flux (float or None)
            - mfcc_1 to mfcc_13 (float or None): Mean MFCC coefficients 1 through 13.

        Returns None if the MBID is invalid, the API request fails, or data is missing.
'''
def get_low_level_audio_features_from_acoustic_brainz(mbid: str) -> Optional[Dict[str, object]]:
    try:
        url = f"{ACOUSTIC_BRAINZ_BASE_URL}/api/v1/{mbid}/low-level"
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        features = {
            "key_key": data.get("tonal", {}).get("key_key"),
            "key_scale": data.get("tonal", {}).get("key_scale"),
            "chords_key": data.get("tonal", {}).get("chords_key"),
            "chords_scale": data.get("tonal", {}).get("chords_scale"),
            "chords_changes_rate": data.get("tonal", {}).get("chords_changes_rate"),
            "chords_number_rate": data.get("tonal", {}).get("chords_number_rate"),
            "bpm": data.get("rhythm", {}).get("bpm"),
            "beats_count": data.get("rhythm", {}).get("beats_count"),
            "average_loudness": data.get("lowlevel", {}).get("average_loudness")
        }

        # Add dynamic, spectral, and MFCC features
        features["dynamic_complexity"] = data.get("lowlevel", {}).get("dynamic_complexity")
        features["duration"] = data.get("metadata", {}).get("audio_properties", {}).get("length")

        spectral = data.get("lowlevel", {})
        features["spectral_centroid"] = spectral.get("spectral_centroid", {}).get("mean")
        features["spectral_rolloff"] = spectral.get("spectral_rolloff", {}).get("mean")
        features["spectral_flux"] = spectral.get("spectral_flux", {}).get("mean")

        # Add MFCCs (mean values only — 13 coefficients)
        mfcc = spectral.get("mfcc", {}).get("mean", [])
        for i in range(min(len(mfcc), 13)):
            features[f"mfcc_{i+1}"] = mfcc[i]

        return features

    except Exception as e:
        print(f"Error while fetching low-level features for MBID {mbid}: {e}")
        return None

'''
    Joins a list of metadata tag values into a comma-separated string.

    This function is used to normalize metadata tags from AcousticBrainz,
    which often return lists of strings even when only one value exists.
    If the input list is empty or None, the function returns None.

    Args:
        tag_list (list): A list of strings representing metadata tag values.

    Returns:
        str or None: A comma-separated string of values if the list is non-empty,
                     otherwise None.
'''
def join_tags(tag_list):
    return ", ".join(tag_list) if tag_list else None

'''
    Fetches and combines high-level and low-level audio features from AcousticBrainz for a given MusicBrainz recording ID (MBID).

    This method performs two separate API calls:
    - One to retrieve high-level audio features (genre, mood, timbre, etc.).
    - Another to retrieve low-level audio features (chords, BPM, loudness, MFCCs, etc.).

    If both calls succeed, the results are merged into a single dictionary that includes the MBID.

    Args:
        mbid (str): The MusicBrainz recording ID of the track.

    Returns:
        Optional[Dict[str, object]]: A dictionary containing merged high-level and low-level audio features with the following keys:
            - mbid (str): The MusicBrainz recording ID.
            - [various]: Keys from both high-level and low-level feature sets (see individual method docs for details).
        Returns None if either set of features is unavailable.
'''
def fetch_audio_features_from_acoustic_brainz(mbid: str) -> Optional[Dict[str, object]]:
    row_features = {}
    row_features["mbid"] = mbid

    # Fetch high-level features
    high_level_features = get_high_level_audio_features_from_acoustic_brainz(mbid)
    if not high_level_features:
        return None
    row_features.update(high_level_features)

    # Respect rate limits before fetching low-level features
    time.sleep(1.0)

    # Fetch low-level features
    low_level_features = get_low_level_audio_features_from_acoustic_brainz(mbid)
    if not low_level_features:
        return None
    row_features.update(low_level_features)

    return row_features