"""Build one ranked list of DJ-compatible song pairs from saved analyses."""

import argparse
import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))

ANALYSIS_FILENAME = "analysis.json"
METADATA_FILENAME = "metadata.json"
JSON_OUTPUT_FILENAME = "compatability_list.json"
CSV_OUTPUT_FILENAME = "compatability_list.csv"
MIN_ANALYSIS_SCHEMA_VERSION = 3
REQUIRED_ANALYSIS_SOURCE = "original_mix"

BPM_TOLERANCE = 5.0
ENERGY_MAX_DELTA = 0.20
ENERGY_COMPONENT_MAX_DELTAS = {
    "loudness": 0.25,
    "dynamic_range": 0.35,
    "timbre": 0.30,
    "onset_rate": 0.30,
    "entropy": 0.20,
}

COMPATIBILITY_WEIGHTS = {
    "phrase_confidence": 0.30,
    "key_confidence": 0.25,
    "popularity": 0.15,
    "energy": 0.30,
}

# Number = Camelot wheel position; A = minor and B = major.
CAMELOT_WHEEL = {
    ("C", "major"): "8B", ("A", "minor"): "8A",
    ("G", "major"): "9B", ("E", "minor"): "9A",
    ("D", "major"): "10B", ("B", "minor"): "10A",
    ("A", "major"): "11B", ("F#", "minor"): "11A",
    ("E", "major"): "12B", ("C#", "minor"): "12A",
    ("B", "major"): "1B", ("G#", "minor"): "1A",
    ("F#", "major"): "2B", ("D#", "minor"): "2A",
    ("Gb", "major"): "2B", ("Eb", "minor"): "2A",
    ("Db", "major"): "3B", ("Bb", "minor"): "3A",
    ("C#", "major"): "3B", ("A#", "minor"): "3A",
    ("Ab", "major"): "4B", ("F", "minor"): "4A",
    ("G#", "major"): "4B",
    ("Eb", "major"): "5B", ("C", "minor"): "5A",
    ("D#", "major"): "5B",
    ("Bb", "major"): "6B", ("G", "minor"): "6A",
    ("A#", "major"): "6B",
    ("F", "major"): "7B", ("D", "minor"): "7A",
}


def clamp01(value):
    """Clamps a numeric value to the inclusive zero-to-one range."""
    return max(0.0, min(1.0, float(value)))


def geometric_pair_score(value_a, value_b):
    """Combines two normalized qualities while penalizing a weak member."""
    return math.sqrt(clamp01(value_a) * clamp01(value_b))


def camelot_for(key, scale):
    """Returns Camelot notation for a tonic and major/minor scale."""
    return CAMELOT_WHEEL.get((key, (scale or "").lower()))


def camelot_number_letter(code):
    """Splits a Camelot code into its wheel number and A/B mode letter."""
    if not code:
        return None, None
    return int(code[:-1]), code[-1]


def keys_are_synced(key_a, scale_a, key_b, scale_b):
    """Checks same, adjacent, or relative major/minor Camelot compatibility."""
    code_a = camelot_for(key_a, scale_a)
    code_b = camelot_for(key_b, scale_b)
    if code_a is None or code_b is None:
        return False
    if code_a == code_b:
        return True

    number_a, letter_a = camelot_number_letter(code_a)
    number_b, letter_b = camelot_number_letter(code_b)
    wheel_distance = abs(number_a - number_b)
    adjacent = letter_a == letter_b and wheel_distance in {1, 11}
    relative = number_a == number_b and letter_a != letter_b
    return adjacent or relative


def find_song_audio(song_dir):
    """Returns the single canonical top-level WAV in a song directory."""
    wavs = sorted(song_dir.glob("*.wav"))
    return wavs[0] if len(wavs) == 1 else None


def load_json(path):
    """Loads one UTF-8 JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_song_analyses(data_dir: Path):
    """Loads analysis, popularity, and file paths for every valid song."""
    songs = []
    required_analysis = {
        "tempo",
        "song_key",
        "song_scale",
        "song_key_strength",
        "overall_phrase_confidence",
        "energy",
        "energy_components",
    }
    required_components = set(ENERGY_COMPONENT_MAX_DELTAS)

    for song_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        analysis_path = song_dir / ANALYSIS_FILENAME
        if not analysis_path.is_file():
            continue
        analysis = load_json(analysis_path)
        if analysis.get("status") != "complete":
            print(f"[warn] {song_dir.name}: analysis is incomplete; skipping")
            continue
        if analysis.get("analysis_source") != REQUIRED_ANALYSIS_SOURCE:
            print(f"[warn] {song_dir.name}: analysis is not original-mix v3; skipping")
            continue
        if int(analysis.get("schema_version", 0)) < MIN_ANALYSIS_SCHEMA_VERSION:
            print(f"[warn] {song_dir.name}: analysis schema is too old; skipping")
            continue
        missing = sorted(required_analysis - set(analysis))
        if missing:
            print(f"[warn] {song_dir.name}: missing analysis fields {missing}; skipping")
            continue

        energy_components = analysis.get("energy_components", {})
        missing_components = sorted(required_components - set(energy_components))
        if missing_components:
            print(f"[warn] {song_dir.name}: missing energy components {missing_components}; skipping")
            continue

        audio_path = find_song_audio(song_dir)
        if audio_path is None:
            print(f"[warn] {song_dir.name}: expected exactly one top-level WAV; skipping")
            continue
        if analysis_path.stat().st_mtime_ns < audio_path.stat().st_mtime_ns:
            print(f"[warn] {song_dir.name}: analysis is older than its source WAV; skipping")
            continue

        metadata_path = song_dir / METADATA_FILENAME
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        songs.append({
            "song_id": song_dir.name,
            "song_path": str(audio_path.resolve()),
            "song_directory": str(song_dir.resolve()),
            "analysis_path": str(analysis_path.resolve()),
            "metadata_path": str(metadata_path.resolve()) if metadata_path.is_file() else None,
            "analysis_schema": int(analysis["schema_version"]),
            "analysis_model": analysis["model_version"],
            "analysis_source": analysis["analysis_source"],
            "tempo": float(analysis["tempo"]),
            "song_key": analysis["song_key"],
            "song_scale": analysis["song_scale"],
            "camelot_key": camelot_for(analysis["song_key"], analysis["song_scale"]),
            "key_confidence": clamp01(analysis["song_key_strength"]),
            "phrase_confidence": clamp01(analysis["overall_phrase_confidence"]),
            "energy": clamp01(analysis["energy"]),
            "energy_components": {
                name: clamp01(energy_components[name])
                for name in ENERGY_COMPONENT_MAX_DELTAS
            },
            "popularity": clamp01(float(metadata.get("popularity", 0.0)) / 100.0),
            "popularity_score": float(metadata.get("popularity", 0.0)),
        })
    return songs


def bpm_compatible(song_a, song_b):
    """Checks whether two songs are within the accepted BPM window."""
    return abs(song_a["tempo"] - song_b["tempo"]) <= BPM_TOLERANCE


def energy_component_differences(song_a, song_b):
    """Returns absolute differences for overall energy and each energy component."""
    differences = {
        name: abs(song_a["energy_components"][name] - song_b["energy_components"][name])
        for name in ENERGY_COMPONENT_MAX_DELTAS
    }
    differences["overall"] = abs(song_a["energy"] - song_b["energy"])
    return differences


def energy_compatible(song_a, song_b):
    """Hard-filters songs whose overall or component energy profiles are too different."""
    differences = energy_component_differences(song_a, song_b)
    if differences["overall"] > ENERGY_MAX_DELTA:
        return False
    return all(
        differences[name] <= max_delta
        for name, max_delta in ENERGY_COMPONENT_MAX_DELTAS.items()
    )


def hard_filters_pass(song_a, song_b):
    """Applies BPM, Camelot key, and rough energy-profile eligibility gates."""
    return (
        bpm_compatible(song_a, song_b)
        and keys_are_synced(
            song_a["song_key"],
            song_a["song_scale"],
            song_b["song_key"],
            song_b["song_scale"],
        )
        and energy_compatible(song_a, song_b)
    )


def energy_compatibility_score(song_a, song_b):
    """Scores closeness within the already-passed overall and component energy gates."""
    differences = energy_component_differences(song_a, song_b)
    component_scores = [
        1.0 - differences[name] / max_delta
        for name, max_delta in ENERGY_COMPONENT_MAX_DELTAS.items()
    ]
    overall_score = 1.0 - differences["overall"] / ENERGY_MAX_DELTA
    return clamp01(0.40 * overall_score + 0.60 * sum(component_scores) / len(component_scores))


def compute_compatibility_components(song_a, song_b):
    """Computes the normalized inputs used to rank one eligible song pair."""
    return {
        "phrase_confidence": geometric_pair_score(
            song_a["phrase_confidence"], song_b["phrase_confidence"],
        ),
        "key_confidence": geometric_pair_score(
            song_a["key_confidence"], song_b["key_confidence"],
        ),
        "popularity": geometric_pair_score(
            song_a["popularity"], song_b["popularity"],
        ),
        "energy": energy_compatibility_score(song_a, song_b),
    }


def compute_compatibility_score(song_a, song_b):
    """Returns the weighted zero-to-one compatibility score for an eligible pair."""
    components = compute_compatibility_components(song_a, song_b)
    score = sum(
        COMPATIBILITY_WEIGHTS[name] * components[name]
        for name in COMPATIBILITY_WEIGHTS
    )
    return clamp01(score), components


def public_song_fields(song):
    """Returns serializable song fields included in each output pair."""
    return {
        "song_id": song["song_id"],
        "song_path": song["song_path"],
        "song_directory": song["song_directory"],
        "analysis_path": song["analysis_path"],
        "metadata_path": song["metadata_path"],
        "analysis_schema": song["analysis_schema"],
        "analysis_model": song["analysis_model"],
        "analysis_source": song["analysis_source"],
        "tempo": song["tempo"],
        "song_key": song["song_key"],
        "song_scale": song["song_scale"],
        "camelot_key": song["camelot_key"],
        "key_confidence": song["key_confidence"],
        "phrase_confidence": song["phrase_confidence"],
        "energy": song["energy"],
        "energy_components": song["energy_components"],
        "popularity_score": song["popularity_score"],
    }


def build_compatibility_list(songs):
    """Builds unique eligible song pairs sorted from most to least compatible."""
    compatible_pairs = []
    for song_a, song_b in combinations(songs, 2):
        if not hard_filters_pass(song_a, song_b):
            continue

        score, score_components = compute_compatibility_score(song_a, song_b)
        energy_differences = energy_component_differences(song_a, song_b)
        compatible_pairs.append({
            "compatibility_score": score,
            "score_components": score_components,
            "hard_filter_metrics": {
                "bpm_difference": abs(song_a["tempo"] - song_b["tempo"]),
                "keys_synced": True,
                "energy_differences": energy_differences,
            },
            "song_a": public_song_fields(song_a),
            "song_b": public_song_fields(song_b),
        })

    compatible_pairs.sort(
        key=lambda pair: (
            pair["compatibility_score"],
            pair["score_components"]["phrase_confidence"],
            pair["score_components"]["key_confidence"],
        ),
        reverse=True,
    )
    for rank, pair in enumerate(compatible_pairs, start=1):
        pair["rank"] = rank
    return compatible_pairs


def write_json_output(pairs, output_path):
    """Writes the ranked compatibility list and filter configuration as JSON."""
    payload = {
        "compatible_mix_count": len(pairs),
        "hard_filters": {
            "bpm_tolerance": BPM_TOLERANCE,
            "energy_max_delta": ENERGY_MAX_DELTA,
            "energy_component_max_deltas": ENERGY_COMPONENT_MAX_DELTAS,
            "key_sync": "same, adjacent, or relative Camelot key",
        },
        "score_weights": COMPATIBILITY_WEIGHTS,
        "analysis_contract": {
            "minimum_schema_version": MIN_ANALYSIS_SCHEMA_VERSION,
            "analysis_source": REQUIRED_ANALYSIS_SOURCE,
        },
        "compatible_mixes": pairs,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def csv_row(pair):
    """Flattens one ranked pair into a CSV-compatible dictionary."""
    song_a = pair["song_a"]
    song_b = pair["song_b"]
    components = pair["score_components"]
    energy_differences = pair["hard_filter_metrics"]["energy_differences"]
    row = {
        "rank": pair["rank"],
        "compatibility_score": pair["compatibility_score"],
        "phrase_confidence_score": components["phrase_confidence"],
        "key_confidence_score": components["key_confidence"],
        "popularity_score": components["popularity"],
        "energy_compatibility_score": components["energy"],
        "bpm_difference": pair["hard_filter_metrics"]["bpm_difference"],
        "overall_energy_difference": energy_differences["overall"],
    }
    for label, song in (("song_a", song_a), ("song_b", song_b)):
        row.update({
            f"{label}_id": song["song_id"],
            f"{label}_path": song["song_path"],
            f"{label}_directory": song["song_directory"],
            f"{label}_analysis_path": song["analysis_path"],
            f"{label}_metadata_path": song["metadata_path"],
            f"{label}_analysis_schema": song["analysis_schema"],
            f"{label}_analysis_model": song["analysis_model"],
            f"{label}_analysis_source": song["analysis_source"],
            f"{label}_tempo": song["tempo"],
            f"{label}_key": song["song_key"],
            f"{label}_scale": song["song_scale"],
            f"{label}_camelot": song["camelot_key"],
            f"{label}_key_confidence": song["key_confidence"],
            f"{label}_phrase_confidence": song["phrase_confidence"],
            f"{label}_energy": song["energy"],
            f"{label}_popularity": song["popularity_score"],
        })
    for name in ENERGY_COMPONENT_MAX_DELTAS:
        row[f"{name}_difference"] = energy_differences[name]
    return row


def write_csv_output(pairs, output_path):
    """Writes the same ranked compatibility list in flattened CSV form."""
    rows = [csv_row(pair) for pair in pairs]
    fieldnames = list(rows[0]) if rows else [
        "rank",
        "compatibility_score",
        "phrase_confidence_score",
        "key_confidence_score",
        "popularity_score",
        "energy_compatibility_score",
        "bpm_difference",
        "overall_energy_difference",
        "song_a_id",
        "song_a_path",
        "song_b_id",
        "song_b_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    """Loads songs, ranks all eligible pairs, and writes only JSON and CSV lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Persistent song-library directory (default: project data directory)",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    songs = load_song_analyses(data_dir)
    pairs = build_compatibility_list(songs)
    json_path = data_dir / JSON_OUTPUT_FILENAME
    csv_path = data_dir / CSV_OUTPUT_FILENAME
    write_json_output(pairs, json_path)
    write_csv_output(pairs, csv_path)
    print(
        f"[pairing] loaded {len(songs)} songs; wrote {len(pairs)} compatible mixes "
        f"to {json_path} and {csv_path}"
    )


if __name__ == "__main__":
    main()
