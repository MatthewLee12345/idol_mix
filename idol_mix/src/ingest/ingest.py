import argparse
import re
import shutil
import base64
import os
import requests
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_PLAYLIST_ID = os.environ.get("SPOTIFY_PLAYLIST_ID", "6kbzPEHj3uMPRFsR3v6xzE")
DEFAULT_MARKET = os.environ.get("SPOTIFY_MARKET", "KR")



def get_access_token(client_id: str, client_secret: str) -> str:
    auth_str = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    r = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_playlist_tracks(token: str, playlist_id: str, market: Optional[str] = None) -> List[Dict]:
    headers = {"Authorization": f"Bearer {token}"}
    fields = (
        "items(added_at,is_local,"
        "track(id,name,uri,popularity,duration_ms,explicit,track_number,disc_number,"
        "is_playable,preview_url,external_urls.spotify,external_ids,"
        "artists(id,name,uri,external_urls.spotify),"
        "album(id,name,release_date,release_date_precision,images,external_urls.spotify)))"
        ",next"
    )

    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    params = {"fields": fields, "limit": 100}
    if market:
        params["market"] = market

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Spotify playlist fetch failed: {r.status_code} {r.text}")
    data = r.json()

    items = data.get("items", [])
    next_url = data.get("next")

    while next_url:
        r = requests.get(next_url, headers=headers, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Spotify pagination failed: {r.status_code} {r.text}")
        page = r.json()
        items.extend(page.get("items", []))
        next_url = page.get("next")

    return items


def flatten_track_item(item: Dict) -> Optional[Dict]:
    track = item.get("track")
    if not track:
        return None

    artists = track.get("artists", [])
    album = track.get("album", {})
    external_ids = track.get("external_ids", {})

    return {
        "playlist_added_at": item.get("added_at"),
        "is_local": item.get("is_local"),
        "track_name": track.get("name"),
        "track_id": track.get("id"),
        "track_uri": track.get("uri"),
        "spotify_url": (track.get("external_urls") or {}).get("spotify"),
        "artist_names": ", ".join(a.get("name", "") for a in artists),
        "artist_ids": ", ".join(a.get("id", "") for a in artists if a.get("id")),
        "artist_uris": ", ".join(a.get("uri", "") for a in artists if a.get("uri")),
        "album_name": album.get("name"),
        "album_id": album.get("id"),
        "album_spotify_url": (album.get("external_urls") or {}).get("spotify"),
        "release_date": album.get("release_date"),
        "release_date_precision": album.get("release_date_precision"),
        "duration_ms": track.get("duration_ms"),
        "duration_sec": round((track.get("duration_ms") or 0) / 1000, 2),
        "explicit": track.get("explicit"),
        "popularity": track.get("popularity"),
        "track_number": track.get("track_number"),
        "disc_number": track.get("disc_number"),
        "is_playable": track.get("is_playable"),
        "preview_url": track.get("preview_url"),
        "isrc": external_ids.get("isrc"),
        "album_image_url": ((album.get("images") or [{}])[0]).get("url"),
        "album_images": album.get("images"),
    }


def sanitize_filename(name: str) -> str:
    if not name:
        name = "unknown"
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")
    return name[:150] if name else "unknown"


def get_song_folder_name(row: Dict) -> str:
    artist = sanitize_filename(row.get("artist_names") or "Unknown Artist")
    title = sanitize_filename(row.get("track_name") or "Unknown Title")
    return f"{title} - {artist}"


def download_file(url: str, dest_path: Path) -> bool:
    if not url:
        return False
    partial_path = dest_path.with_name(f".{dest_path.name}.part")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(partial_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        partial_path.replace(dest_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        partial_path.unlink(missing_ok=True)
        return False


def get_best_album_image_url(row: Dict) -> Optional[str]:
    images = row.get("album_images") or []
    if images:
        images_sorted = sorted(images, key=lambda im: (im.get("width") or 0), reverse=True)
        return images_sorted[0].get("url")
    return row.get("album_image_url")


def ensure_album_cover(row: Dict, song_dir: Path) -> bool:
    """Keep an existing cover or download it when a daily run finds it missing."""
    existing = [
        path for path in song_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if existing:
        return True
    cover_url = get_best_album_image_url(row)
    if not cover_url:
        print(f"No album cover found for: {song_dir.name}")
        return False
    lower_url = cover_url.lower()
    extension = ".png" if ".png" in lower_url else ".jpeg" if ".jpeg" in lower_url else ".jpg"
    cover_path = song_dir / f"cover{extension}"
    if download_file(cover_url, cover_path):
        print(f"Cover saved: {cover_path}")
        return True
    return False


def download_wav(spotify_url: str, song_dir: Path) -> bool:
    """Attempt to download the wav via spotdl. Returns True only if a .wav
    file actually exists in song_dir afterward."""
    cmd = [
        "spotdl",
        spotify_url,
        "--format", "wav",
        "--output", str(song_dir / "{artist} - {title}.{output-ext}")
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdin=subprocess.DEVNULL,
            timeout=1800,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"spotdl failed: {e}")
        return False

    wav_files = list(song_dir.glob("*.wav"))
    return len(wav_files) > 0


def load_existing_compiled(json_path: Path) -> Dict[str, Dict]:
    """
    Load existing compiled metadata and key it by track_id so repeated
    ingest runs can skip already-downloaded songs while still updating
    metadata like popularity.
    """
    if not json_path.is_file():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        existing_rows = json.load(f)

    existing_by_id = {}
    for row in existing_rows:
        track_id = row.get("track_id")
        if track_id:
            existing_by_id[track_id] = row

    return existing_by_id


def find_song_dir_by_track_id(data_dir: Path, track_id: str) -> Optional[Path]:
    """Resolve an existing song directory by its stable Spotify track ID."""
    matches = []
    for song_dir in data_dir.iterdir():
        metadata_path = song_dir / "metadata.json"
        if not song_dir.is_dir() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("track_id") == track_id:
            matches.append(song_dir)
    if len(matches) > 1:
        raise RuntimeError(f"Track ID {track_id} exists in multiple song directories")
    return matches[0] if matches else None


def update_song_folder_metadata(song_dir: Path, row: Dict) -> None:
    metadata_path = song_dir / "metadata.json"
    temporary_path = metadata_path.with_name(".metadata.json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    temporary_path.replace(metadata_path)


def write_compiled_files(all_rows: List[Dict], json_path: Path, csv_path: Path) -> None:
    temporary_json = json_path.with_name(f".{json_path.name}.tmp")
    with open(temporary_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    temporary_json.replace(json_path)

    if not all_rows:
        return

    fieldnames = list(all_rows[0].keys())
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with open(temporary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    temporary_csv.replace(csv_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest a Spotify playlist into the song library")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--playlist-id", default=DEFAULT_PLAYLIST_ID)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    args = parser.parse_args(argv)

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be provided by the environment"
        )
    if not args.playlist_id:
        raise RuntimeError("A Spotify playlist ID is required")

    token = get_access_token(client_id, client_secret)
    items = get_playlist_tracks(token, args.playlist_id, args.market or None)
    rows = [r for r in (flatten_track_item(item) for item in items) if r is not None]

    if not rows:
        raise RuntimeError("No tracks were returned from the playlist.")

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = data_dir / "compiled_metadata.json"
    summary_csv_path = data_dir / "compiled_metadata.csv"

    existing_by_id = load_existing_compiled(summary_json_path)

    new_download_count = 0
    updated_existing_count = 0
    skipped_existing_downloads = 0
    failed_tracks = []

    current_playlist_by_id: Dict[str, Dict] = {}

    for row in rows:
        track_id = row.get("track_id")
        spotify_url = row.get("spotify_url")
        track_label = row.get("track_name") or "Unknown Title"

        if not track_id:
            print(f"Skipping (no track_id): {track_label}")
            failed_tracks.append(track_label)
            continue

        if not spotify_url:
            print(f"Skipping (no spotify_url): {track_label}")
            failed_tracks.append(track_id)
            continue

        already_known = track_id in existing_by_id
        folder_name = get_song_folder_name(row)
        known_song_dir = find_song_dir_by_track_id(data_dir, track_id)
        song_dir = known_song_dir or data_dir / folder_name
        if known_song_dir is None and song_dir.is_dir():
            existing_metadata_path = song_dir / "metadata.json"
            existing_track_id = None
            if existing_metadata_path.is_file():
                try:
                    with existing_metadata_path.open("r", encoding="utf-8") as handle:
                        existing_track_id = json.load(handle).get("track_id")
                except (OSError, json.JSONDecodeError):
                    pass
            if existing_track_id and existing_track_id != track_id:
                song_dir = data_dir / f"{folder_name} [{track_id}]"

        if already_known and any(song_dir.glob("*.wav")):
            skipped_existing_downloads += 1
            song_dir.mkdir(parents=True, exist_ok=True)
            update_song_folder_metadata(song_dir, row)
            if not ensure_album_cover(row, song_dir):
                failed_tracks.append(track_id)
            current_playlist_by_id[track_id] = row
            updated_existing_count += 1
            print(f"Metadata refreshed, download skipped: {folder_name}")
            continue

        song_dir_preexisting = song_dir.exists()
        song_dir.mkdir(parents=True, exist_ok=True)

        wav_ok = download_wav(spotify_url, song_dir)
        if not wav_ok:
            print(f"WAV download failed, skipping entirely: {folder_name}")
            failed_tracks.append(track_id)
            if not song_dir_preexisting:
                shutil.rmtree(song_dir, ignore_errors=True)
            continue

        print(f"Song downloaded: {folder_name}")

        update_song_folder_metadata(song_dir, row)

        if not ensure_album_cover(row, song_dir):
            failed_tracks.append(track_id)

        current_playlist_by_id[track_id] = row
        new_download_count += 1

    merged_by_id = dict(existing_by_id)

    for track_id, fresh_row in current_playlist_by_id.items():
        merged_by_id[track_id] = fresh_row

    merged_rows = list(merged_by_id.values())
    merged_rows.sort(key=lambda x: (x.get("popularity") is None, -(x.get("popularity") or -1)))

    write_compiled_files(merged_rows, summary_json_path, summary_csv_path)

    print(
        f"Finished ingest. New downloads: {new_download_count}, "
        f"existing songs refreshed: {updated_existing_count}, "
        f"downloads skipped: {skipped_existing_downloads}, "
        f"compiled total: {len(merged_rows)}"
    )
    print(f"Updated summary index: {summary_json_path}, {summary_csv_path}")
    if failed_tracks:
        raise RuntimeError(
            f"Ingest completed with {len(set(failed_tracks))} failed track(s): "
            + ", ".join(sorted(set(failed_tracks)))
        )


if __name__ == "__main__":
    main()
