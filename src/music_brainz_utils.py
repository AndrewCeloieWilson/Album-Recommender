import random
import requests
from typing import Dict, Optional

from config import (
    MUSIC_BRAINZ_BASE_URL,
    HEADERS,
)

'''
    Retrieves metadata features for a given recording MBID using the MusicBrainz API.

    Args:
        mbid (str): MusicBrainz ID for the recording.

    Returns:
        Optional[Dict[str, Optional[object]]]: Dictionary of metadata features with keys:
            - 'release_year' (int or None)
            - 'release_title' (str or None)
            - 'recording_title' (str or None)
          Returns None if the HTTP request fails or an error occurs.
'''
def get_metadata_features_from_music_brainz(mbid: str) -> Optional[Dict[str, Optional[object]]]:
    try:
        url = f"{MUSIC_BRAINZ_BASE_URL}/ws/2/recording/{mbid}?fmt=json&inc=releases+artists"
        response = requests.get(url, headers=HEADERS, timeout=10)

        features = {
            "release_year": None,
            "release_title": None,
            "recording_title": None
        }

        if response.status_code == 200:
            data = response.json()

            features["recording_title"] = data.get("title")
            releases = data.get("releases", [])
            if releases:
                # Sort the releases by their release date in ascending order
                # Use a far-future default date ("9999-12-31") for any release missing a date
                sorted_releases = sorted(releases, key=lambda r: r.get("date", "9999-12-31"))
            
                # Select the earliest release after sorting
                first_release = sorted_releases[0]
            
                # Capture the title of the earliest release
                features["release_title"] = first_release.get("title")
            
                # Extract the release date string from the earliest release
                date = first_release.get("date")
            
                # If the date is present and at least 4 characters long, parse the year portion
                if date and len(date) >= 4:
                    # Convert the first 4 characters of the date (the year) to an integer
                    features["release_year"] = int(date[:4])

            return features
        else:
            print(f"Warning: Failed to fetch metadata for recording {mbid} — HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"Error while fetching low-level features for MBID {mbid}: {e}")
        return None

'''
    Fetches a small batch of random album releases from the MusicBrainz API.

    This function issues a GET request to the MusicBrainz `/ws/2/release` endpoint with a wildcard
    query and a random offset, returning a batch of album-level metadata.

    Args:
        batch_size (int, optional): Number of album releases to return. Defaults to 10.

    Returns:
        Optional[list[dict]]: A list of MusicBrainz release records, where each dictionary contains
                              metadata about an album release. Returns None if the request fails.
'''
def fetch_random_album_batch(batch_size: int = 10) -> Optional[list[dict]]:
    try:
        offset = random.randint(0, 4500000)  # Start at a random offset for variety
        url = f"{MUSIC_BRAINZ_BASE_URL}/ws/2/release?query=*&fmt=json&limit={batch_size}&offset={offset}"
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            return response.json().get('releases', [])
        else:
            print(f"Warning: Failed to fetch releases — HTTP {response.status_code}")
            return None

    except requests.RequestException as e:
        print(f"Error fetching random album batch: {e}")
        return None

'''
    Retrieves all track MusicBrainz IDs (MBIDs) and titles for a given album release ID.

    This function queries the MusicBrainz API for the specified release, extracts all associated
    tracks, and returns their MBIDs and titles. Only recordings with valid IDs are included.

    Args:
        release_id (str): The MusicBrainz release ID for the album.

    Returns:
        Optional[list[tuple[str, str]]]: A list of (recording MBID, track title) pairs.
                                         Returns None if the request fails or no tracks are found.
'''
def get_album_tracks(release_id: str) -> Optional[list[tuple[str, str]]]:
    try:
        url = f"{MUSIC_BRAINZ_BASE_URL}/ws/2/release/{release_id}?inc=recordings&fmt=json"
        response = requests.get(url, headers=HEADERS, timeout=10)

        mbid_list = []
        if response.status_code == 200:
            media = response.json().get('media', [])
            for m in media:
                for track in m.get('tracks', []):
                    recording = track.get('recording', {})
                    if 'id' in recording:
                        mbid_list.append((recording['id'], track.get('title', 'Unknown')))
        else:
            print(f"Warning: Failed to fetch album tracks for release {release_id} — HTTP {response.status_code}")
            return None
        return mbid_list

    except requests.RequestException as e:
        print(f"Error fetching tracks for album {release_id}: {e}")
        return None