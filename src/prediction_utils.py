# Standard library imports
import os
import sys
import time
from typing import Optional, Dict, Any, Generator, Tuple, List

# Add the parent directory of 'notebooks' to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Local application imports
from src.acoustic_brainz_utils import fetch_audio_features_from_acoustic_brainz
from src.music_brainz_utils import get_metadata_features_from_music_brainz, get_album_tracks, fetch_random_album_batch

'''
    Continuously fetches random albums and yields those with at least a minimum number of tracks.
    
    This generator queries MusicBrainz for random album releases and filters for those 
    meeting the minimum track count. If they qualify, it yields album metadata and associated tracks.
    
    Args:
        track_length_threshold (int): Minimum number of tracks an album must have.
    
    Yields:
        Tuple[dict, List[Tuple[str, str]]]: 
            - Album metadata dictionary from MusicBrainz.
            - List of (track MBID, track title) tuples for that album.
'''
def fetch_valid_albums_for_prediction(
    track_length_threshold: int
) -> Generator[Tuple[dict, List[Tuple[str, str]]], None, None]:
    while True:
        albums = fetch_random_album_batch()
        time.sleep(1.0)  # API rate limiting
        if not albums:
            continue

        for album in albums:
            tracks = get_album_tracks(album['id'])
            if tracks and len(tracks) >= track_length_threshold:
                yield album, tracks
                
'''
    Fetches and combines audio and metadata features for a given track MBID. Returns none if 
    acoustic brainz features, music brainz features, or release year cannot be retreived.

    Args:
        mbid (str): The MusicBrainz Identifier for the track.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of combined features if successful, or None if
        required data (e.g., release year) is missing.
'''
def gather_features_for_prediction(mbid: str) -> Optional[Dict[str, Any]]:
    time.sleep(1.0)  # Respect API rate limits
    acoustic = fetch_audio_features_from_acoustic_brainz(mbid)
    if not acoustic:
        return None

    time.sleep(1.0)
    metadata = get_metadata_features_from_music_brainz(mbid)
    if not metadata or not metadata.get("release_year"):
        # Optional add 'or not metadata.get("release_year")' if you are using release_year in your inputs
        # And want to only predict with data that includes the release_year 
        return None

    return {**acoustic, **metadata}

    