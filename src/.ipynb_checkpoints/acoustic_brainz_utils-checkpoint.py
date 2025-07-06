import requests  # For sending HTTP requests to the AcousticBrainz API
import time

HEADERS = {"User-Agent": "SongRecommenderBot/1.0 (437andrew@gmail.com)"}

'''
    Fetches selected audio features and metadata from the AcousticBrainz high-level endpoint
    using the given MBID.

    This function accesses the AcousticBrainz API, retrieves a JSON response,
    and extracts a set of predefined high-level features and metadata useful for audio 
    classification or recommendation tasks.

    Args:
        mbid (str): The MusicBrainz recording ID of the track.

    Returns:
        dict: A dictionary containing selected high-level audio features and metadata:
            - danceable (float): Probability the track is danceable.
            - mood_* (float): Probabilities for various moods (acoustic, aggressive, happy, etc.).
            - genre_* (float): Probabilities for subgenres across multiple genre classifiers.
            - ismir04_rhythm_* (float): Probabilities for dance rhythm styles.
            - moods_mirex_* (float): Probabilities for MIREX mood clusters.
            - timbre (float): Timbre probability (bright vs. dark).
            - tonal_atonal (float): Tonality probability.
            - voice_instrumental (float): Instrumental probability.
            - album (str): Album name (if available).
            - artist (str): Artist name (if available).
            - genre (str): Genre tag (if available).
            - label (str): Record label (if available).
            - title (str): Track title (if available).

        Returns None if the MBID is not found, the response fails, or required data is missing.
'''
def get_high_level_audio_features_from_acoustic_brainz(mbid):
    try:
        response = requests.get(f"https://acousticbrainz.org/api/v1/{mbid}/high-level", headers= HEADERS)
        
        if response.status_code != 200:
            print(f"Warning: Failed to fetch high-level features for MBID {mbid} — status {response.status_code}")
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

        # Basic metadata from tags
        features["album"] = join_tags(metadata_tags.get("album", []))
        features["artist"] = join_tags(metadata_tags.get("artist", []))
        features["genre"] = join_tags(metadata_tags.get("genre", []))
        features["label"] = join_tags(metadata_tags.get("label", []))
        features["title"] = join_tags(metadata_tags.get("title", []))
        
        return features
        
    except Exception as e:
        print(f"Error while fetching high-level features for MBID {mbid}: {e}")
        return None

'''
    Fetches low-level audio features from AcousticBrainz for a given MusicBrainz recording ID (MBID).
    
    Args:
        mbid (str): The MusicBrainz recording ID of the track.
    
    Returns:
        dict or None: A dictionary containing selected low-level audio features:
            - key_key (str)
            - key_scale (str)
            - chords_key (str)
            - chords_scale (str)
            - chords_changes_rate (float)
            - chords_number_rate (float)
            - bpm (float)
            - beats_count (int)
            - average_loudness (float)
    
        Returns None if the MBID is invalid, data is missing, or an error occurs.
'''
def get_low_level_audio_features_from_acoustic_brainz(mbid):
    try:
        url = f"https://acousticbrainz.org/api/v1/{mbid}/low-level"
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            print(f"Warning: Failed to fetch low-level features for MBID {mbid} — status {response.status_code}")
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
    Fetches high-level and low-level audio features from AcousticBrainz for a given MusicBrainz ID (MBID).

    This method calls the AcousticBrainz API twice—once for high-level features and once for low-level features.
    If both calls succeed, their results are merged into a single dictionary.

    Args:
        mbid (str): The MusicBrainz recording ID to query.

    Returns:
        dict or None: Dictionary containing merged high-level and low-level features along with the MBID. 
        Returns None if high-level features are unavailable.
'''
def fetch_audio_features_from_acoustic_brainz(mbid):
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
    if low_level_features:
        row_features.update(low_level_features)
    else:
        print(f"Warning: Low-level features not found for MBID {mbid}")

    return row_features