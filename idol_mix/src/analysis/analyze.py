"""Sequence-aware K-pop phrase analysis derived only from the original mix."""

import argparse
import json
import math
import os
from pathlib import Path

import essentia
import essentia.standard as estd
import librosa
import madmom
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.sparse.csgraph
import sklearn.cluster
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_FILENAME = "analysis.json"
ANALYSIS_SCHEMA_VERSION = 3
ANALYSIS_MODEL_VERSION = "kpop-sequence-original-mix-v3"
ANALYSIS_SOURCE = "original_mix"

SR = 22050
HOP_LENGTH = 512
N_MFCC = 13
N_CLUSTERS = 6
BEATS_PER_BAR = 4
HYPERBAR_SIZE = 4
MIN_BPM = 70
MAX_BPM = 150
W_DRUMS = 0.40
W_HARMONY = 0.60
MIN_LAG_BARS_FOR_REPETITION = 8
KEY_PROFILE_TYPE = "edma"
KEY_HPCP_SIZE = 36
ENERGY_WEIGHTS = {
    "loudness": 0.30,
    "dynamic_range": 0.15,
    "timbre": 0.20,
    "onset_rate": 0.20,
    "entropy": 0.15,
}
ENERGY_TEMPO_MIN = MIN_BPM
ENERGY_TEMPO_MAX = MAX_BPM

MAIN_PHRASE_LENGTHS = (8, 16, 32)
ALLOWED_SEGMENT_LENGTHS = tuple(range(4, 33))
EDGE_LENGTHS = (1, 2, 4, 8, 16, 32)
TRANSITION_LENGTHS = (1, 2, 3, 4)
PATTERN_SIMILARITY_THRESHOLD = 0.86
TRANSITION_SCORE_THRESHOLD = 0.60
EDGE_SCORE_THRESHOLD = 0.52
SEQUENCE_SLOTS = 8


def build_key_extractor():
    """Builds a shared Essentia HPCP and key-estimation pipeline."""
    windowing = estd.Windowing(type="blackmanharris62")
    spectrum = estd.Spectrum()
    spectral_peaks = estd.SpectralPeaks(
        orderBy="magnitude", magnitudeThreshold=1e-5, minFrequency=20,
        maxFrequency=3500, maxPeaks=60,
    )
    hpcp = estd.HPCP(
        size=KEY_HPCP_SIZE, referenceFrequency=440, bandPreset=False,
        minFrequency=20, maxFrequency=3500, weightType="cosine",
        nonLinear=False, windowSize=1.0,
    )
    key_detector = estd.Key(
        profileType=KEY_PROFILE_TYPE, numHarmonics=4, pcpSize=KEY_HPCP_SIZE,
        slope=0.6, usePolyphony=True, useThreeChords=True,
    )
    return windowing, spectrum, spectral_peaks, hpcp, key_detector


def estimate_key_for_slice(y_slice, sr, pipeline, frame_size=4096, hop_size=2048):
    """Estimates key, scale, and strength for an arbitrary audio slice."""
    windowing, spectrum, spectral_peaks, hpcp, key_detector = pipeline
    if len(y_slice) < frame_size:
        return "N", "none", 0.0
    y_slice = essentia.array(y_slice.astype(np.float32))
    hpcp_frames = []
    for frame in estd.FrameGenerator(
        y_slice, frameSize=frame_size, hopSize=hop_size, startFromZero=True,
    ):
        frequencies, magnitudes = spectral_peaks(spectrum(windowing(frame)))
        hpcp_frames.append(hpcp(frequencies, magnitudes))
    if not hpcp_frames:
        return "N", "none", 0.0
    average_hpcp = essentia.array(np.mean(np.asarray(hpcp_frames), axis=0))
    key, scale, strength, _ = key_detector(average_hpcp)
    return key, scale, float(strength)


def list_song_wavs(data_dir: Path):
    """Lists the single canonical top-level WAV from each song directory."""
    wavs = []
    for song_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        candidates = sorted(song_dir.glob("*.wav"))
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Multiple top-level WAVs in {song_dir}: {names}")
        if candidates:
            wavs.append(candidates[0])
    return wavs


def find_song_wav(song_name: str, data_dir: Path) -> Path:
    """Resolves a directory name or partial song name to its WAV path."""
    target_lower = song_name.strip().lower()
    for wav in list_song_wavs(data_dir):
        directory_name = wav.parent.name.lower()
        if target_lower in directory_name or directory_name in target_lower:
            return wav
    raise FileNotFoundError(f"No song matching '{song_name}' in {data_dir}")


def has_current_analysis(song_dir: Path) -> bool:
    """Return whether a song already has a complete current original-mix analysis."""
    path = song_dir / OUTPUT_FILENAME
    if not path.is_file():
        return False
    wavs = list(song_dir.glob("*.wav"))
    if len(wavs) != 1 or path.stat().st_mtime_ns < wavs[0].stat().st_mtime_ns:
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("status") == "complete"
        and result.get("schema_version") == ANALYSIS_SCHEMA_VERSION
        and result.get("model_version") == ANALYSIS_MODEL_VERSION
        and result.get("analysis_source") == ANALYSIS_SOURCE
    )


def load_mono(path: Path, sr: int = SR):
    """Loads an audio file as a mono signal at the requested sample rate."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio, sr


def compute_loudness_score(y_full):
    """Scores integrated RMS loudness against a fixed commercial-audio range."""
    rms = librosa.feature.rms(y=y_full, hop_length=HOP_LENGTH)[0]
    mean_db = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))
    return float(np.clip((mean_db + 40.0) / 34.0, 0.0, 1.0))


def compute_dynamic_range_score(full_rms_sync):
    """Scores sustained energy by inversely scaling beat-synchronous RMS spread."""
    if len(full_rms_sync) < 2:
        return 0.5
    rms_db = librosa.amplitude_to_db(
        full_rms_sync, ref=np.max(full_rms_sync) + 1e-8,
    )
    return float(1.0 - np.clip(float(np.std(rms_db)) / 20.0, 0.0, 1.0))


def compute_timbre_score(centroid_sync, sr):
    """Scores spectral brightness as the timbral component of energy."""
    if len(centroid_sync) == 0:
        return 0.5
    return float(np.clip(float(np.mean(centroid_sync)) / (sr / 4.0), 0.0, 1.0))


def compute_onset_rate_score(drums_onset_sync, tempo):
    """Combines normalized drum onset density with a bounded tempo prior."""
    if len(drums_onset_sync) == 0:
        onset_density_score = 0.5
    else:
        normalized = drums_onset_sync / (np.max(drums_onset_sync) + 1e-8)
        onset_density_score = float(np.mean(normalized))
    tempo_score = np.clip(
        (tempo - ENERGY_TEMPO_MIN) / (ENERGY_TEMPO_MAX - ENERGY_TEMPO_MIN),
        0.0, 1.0,
    )
    return float(0.7 * onset_density_score + 0.3 * tempo_score)


def compute_entropy_score(y_full):
    """Scores spectral entropy from the full mix's average magnitude spectrum."""
    mean_spectrum = np.mean(np.abs(librosa.stft(y_full, hop_length=HOP_LENGTH)), axis=1)
    probabilities = mean_spectrum / (np.sum(mean_spectrum) + 1e-8)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(np.clip(entropy / (np.log2(len(mean_spectrum)) + 1e-8), 0.0, 1.0))


def compute_energy_score(y_full, sr, feats, tempo):
    """Computes composite energy and its five perceptual component scores."""
    components = {
        "loudness": compute_loudness_score(y_full),
        "dynamic_range": compute_dynamic_range_score(feats["full_rms_sync"]),
        "timbre": compute_timbre_score(feats["centroid_sync"], sr),
        "onset_rate": compute_onset_rate_score(feats["drums_onset_sync"], tempo),
        "entropy": compute_entropy_score(y_full),
    }
    energy = sum(ENERGY_WEIGHTS[name] * value for name, value in components.items())
    return {"energy": float(np.clip(energy, 0.0, 1.0)), "energy_components": components}


def madmom_downbeats(y, sr, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """Detects beats/downbeats and estimates beat-grid regularity confidence."""
    signal = madmom.audio.signal.Signal(y, sr)
    activations = RNNDownBeatProcessor()(signal)
    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=[BEATS_PER_BAR], fps=100, min_bpm=min_bpm, max_bpm=max_bpm,
    )
    beats = tracker(activations)
    if beats.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=int), [], 0.0
    beat_times = beats[:, 0].astype(float)
    bar_index = beats[:, 1].astype(int)
    beat_times, bar_index = correct_octave_error(beat_times, bar_index, sr, y)
    downbeat_times = beat_times[bar_index == 1].tolist()
    if len(beat_times) >= 3:
        intervals = np.diff(beat_times)
        confidence = float(np.clip(
            1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8), 0.0, 1.0,
        ))
    else:
        confidence = 0.0
    return beat_times, bar_index, downbeat_times, confidence


def correct_octave_error(beat_times, bar_index, sr, y):
    """Corrects a likely double-tempo beat grid using an independent tempo estimate."""
    if len(beat_times) < 4:
        return beat_times, bar_index
    madmom_interval = float(np.median(np.diff(beat_times)))
    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo_estimate = librosa.feature.tempo(
            onset_envelope=onset_envelope, sr=sr, aggregate=np.median,
        )
    except AttributeError:
        tempo_estimate = librosa.beat.tempo(
            onset_envelope=onset_envelope, sr=sr, aggregate=np.median,
        )
    if len(tempo_estimate) == 0 or tempo_estimate[0] <= 0:
        return beat_times, bar_index
    if 1.7 <= (60.0 / tempo_estimate[0]) / madmom_interval <= 2.3:
        beat_times = beat_times[::2]
        bar_index = renumber_bar_index(len(beat_times))
    return beat_times, bar_index


def renumber_bar_index(n_beats, beats_per_bar=BEATS_PER_BAR):
    """Builds repeating one-based beat positions within each bar."""
    return (np.arange(n_beats) % beats_per_bar) + 1


def extend_beatgrid_to_duration(beat_times, bar_index, audio_duration, tolerance=0.05):
    """Extends a valid beat grid to cover a trailing portion of the audio."""
    if len(beat_times) < 4:
        return beat_times, bar_index
    gap = audio_duration - beat_times[-1]
    if gap <= tolerance:
        return beat_times, bar_index
    median_interval = float(np.median(np.diff(beat_times[-8:])))
    if median_interval <= 0:
        return beat_times, bar_index
    extra_count = int(np.floor(gap / median_interval))
    if extra_count <= 0:
        return beat_times, bar_index
    offsets = np.arange(1, extra_count + 1)
    extra_times = beat_times[-1] + median_interval * offsets
    extra_bar_indices = ((offsets + bar_index[-1] - 1) % BEATS_PER_BAR) + 1
    return np.concatenate([beat_times, extra_times]), np.concatenate([bar_index, extra_bar_indices])


def beat_frames_from_times(beat_times, sr, n_frames):
    """Converts beat times to unique in-range analysis frame indices."""
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP_LENGTH)
    return np.unique(np.clip(frames, 0, n_frames - 1))


def sync_feature(feature_2d, beat_frames, aggregate=np.mean):
    """Aggregates a frame-level feature between consecutive beat frames."""
    return librosa.util.sync(feature_2d, beat_frames, aggregate=aggregate, pad=False)


def build_mix_beat_sync_features(y_full, sr, beat_times):
    """Builds beat-synchronous features solely from the original mix."""
    n_frames_full = 1 + len(y_full) // HOP_LENGTH
    beat_frames = beat_frames_from_times(beat_times, sr, n_frames_full)
    if len(beat_frames) < BEATS_PER_BAR * HYPERBAR_SIZE:
        return None

    def onset_and_rms(signal):
        """Computes onset strength and RMS series for one signal."""
        onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)[0]
        return onset, rms

    harmonic_audio, percussive_audio = librosa.effects.hpss(y_full)
    drums_audio = percussive_audio
    drums_onset, drums_rms = onset_and_rms(drums_audio)
    cqt = np.abs(librosa.cqt(
        harmonic_audio, sr=sr, hop_length=HOP_LENGTH, bins_per_octave=36, n_bins=252,
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    mfcc = librosa.feature.mfcc(
        y=harmonic_audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
    )
    chroma = librosa.feature.chroma_cqt(
        y=harmonic_audio, sr=sr, hop_length=HOP_LENGTH,
    )
    centroid = librosa.feature.spectral_centroid(
        y=y_full, sr=sr, hop_length=HOP_LENGTH,
    )[0]
    full_magnitude = np.abs(librosa.stft(y_full, hop_length=HOP_LENGTH))
    frequencies = librosa.fft_frequencies(sr=sr)
    bass_bins = frequencies <= 180.0
    bass_rms = np.sqrt(np.mean(np.square(full_magnitude[bass_bins]), axis=0))
    full_rms = librosa.feature.rms(y=y_full, hop_length=HOP_LENGTH)[0]
    full_onset = librosa.onset.onset_strength(y=y_full, sr=sr, hop_length=HOP_LENGTH)
    n_frames = min(
        cqt_db.shape[1], mfcc.shape[1], chroma.shape[1], len(drums_onset),
        len(drums_rms), len(bass_rms), len(full_rms),
        len(full_onset), len(centroid),
    )
    beat_frames = beat_frames[beat_frames < n_frames]
    if len(beat_frames) < BEATS_PER_BAR * HYPERBAR_SIZE:
        return None

    cqt_sync = sync_feature(cqt_db[:, :n_frames], beat_frames, aggregate=np.median)
    mfcc_sync = sync_feature(mfcc[:, :n_frames], beat_frames)
    chroma_sync = sync_feature(chroma[:, :n_frames], beat_frames)
    scalar_features = {
        "drums_onset_sync": drums_onset, "drums_rms_sync": drums_rms,
        "bass_rms_sync": bass_rms,
        "full_rms_sync": full_rms, "full_onset_sync": full_onset,
        "centroid_sync": centroid,
    }
    scalar_features = {
        name: sync_feature(values[np.newaxis, :n_frames], beat_frames)[0]
        for name, values in scalar_features.items()
    }
    beat_time_sync = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    length = min(
        cqt_sync.shape[1], mfcc_sync.shape[1], chroma_sync.shape[1],
        len(beat_time_sync), *(len(values) for values in scalar_features.values()),
    )
    return {
        "beat_times": beat_time_sync[:length], "cqt_sync": cqt_sync[:, :length],
        "mfcc_sync": mfcc_sync[:, :length], "chroma_sync": chroma_sync[:, :length],
        **{name: values[:length] for name, values in scalar_features.items()},
    }


def recurrence_from_1d(series, width=3):
    """Builds an affinity recurrence matrix from a one-dimensional feature."""
    return librosa.segment.recurrence_matrix(
        series.reshape(1, -1), width=width, mode="affinity", sym=True,
    )


def fused_affinity_matrix(feats):
    """Fuses original-mix harmonic, rhythmic, and path similarity."""
    n_beats = feats["cqt_sync"].shape[1]
    harmony = librosa.segment.recurrence_matrix(
        feats["cqt_sync"], width=3, mode="affinity", sym=True,
    )
    drums = recurrence_from_1d(feats["drums_onset_sync"] + feats["drums_rms_sync"])
    path_distance = np.sum(np.diff(feats["mfcc_sync"], axis=1) ** 2, axis=0)
    sigma = np.median(path_distance[path_distance > 0]) if np.any(path_distance > 0) else 1.0
    path_similarity = np.exp(-path_distance / (sigma + 1e-8))
    path = np.diag(path_similarity, k=1) + np.diag(path_similarity, k=-1)
    fused = W_HARMONY * harmony + W_DRUMS * drums
    path_degree = np.sum(path, axis=1)
    fused_degree = np.sum(fused, axis=1)
    denominator = np.sum((path_degree + fused_degree) ** 2)
    mix = (
        path_degree.dot(path_degree + fused_degree) / denominator
        if denominator > 0 else 0.5
    )
    return mix * fused + (1.0 - mix) * path, n_beats


def laplacian_segment(affinity, n_beats, k=N_CLUSTERS):
    """Finds coarse structural boundaries by clustering a normalized graph Laplacian."""
    if n_beats < 2:
        return np.asarray([], dtype=int)
    k = min(k, max(2, n_beats // 8))
    laplacian = scipy.sparse.csgraph.laplacian(affinity, normed=True)
    _, eigenvectors = scipy.linalg.eigh(laplacian)
    eigenvectors = scipy.ndimage.median_filter(eigenvectors, size=(9, 1))
    cumulative_norm = np.cumsum(eigenvectors**2, axis=1) ** 0.5
    embedding = eigenvectors[:, :k] / (cumulative_norm[:, k - 1:k] + 1e-8)
    clusterer = sklearn.cluster.KMeans(n_clusters=k, n_init="auto", random_state=0)
    segment_ids = clusterer.fit_predict(embedding)
    boundaries = 1 + np.flatnonzero(segment_ids[:-1] != segment_ids[1:])
    return librosa.util.fix_frames(boundaries, x_min=0, x_max=n_beats - 1)


def nearest_beat_indices(beat_times_sync, target_times):
    """Finds the closest synchronized beat index for each target time."""
    target_times = np.asarray(target_times, dtype=float)
    indices = np.searchsorted(beat_times_sync, target_times)
    indices = np.clip(indices, 1, len(beat_times_sync) - 1)
    left = indices - 1
    left_distance = np.abs(beat_times_sync[left] - target_times)
    right_distance = np.abs(beat_times_sync[indices] - target_times)
    return np.where(left_distance <= right_distance, left, indices)


def compute_bar_level_features(feats, downbeat_times):
    """Builds bar vectors anchored to detected downbeats and their beat ranges."""
    chroma_sync = feats["chroma_sync"]
    beat_times_sync = feats["beat_times"]
    n_beats = chroma_sync.shape[1]
    if len(downbeat_times) < 2 or n_beats < 2:
        return None, None, None
    downbeat_indices = nearest_beat_indices(beat_times_sync, downbeat_times)
    downbeat_indices = np.unique(np.clip(downbeat_indices, 0, n_beats - 1))
    if len(downbeat_indices) < 2:
        return None, None, None
    bar_starts = downbeat_indices[:-1]
    bar_ends = downbeat_indices[1:]
    bar_vectors = np.zeros((len(bar_starts), chroma_sync.shape[0]))
    for index, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        bar_vectors[index] = (
            np.mean(chroma_sync[:, start:end], axis=1)
            if end > start else chroma_sync[:, start]
        )
    return bar_vectors, bar_starts.astype(int), bar_ends.astype(int)

CORE_ROLES = (
    "intro",
    "verse",
    "pre-chorus",
    "chorus",
    "post-chorus",
    "bridge",
    "instrumental",
    "transition",
    "outro",
)

ENERGY_LABELS = ("low", "low-mid", "mid", "high-mid", "high")


def tie_aware_percentiles(values):
    """Ranks values from zero to one while assigning ties the same average rank."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if np.allclose(values, values[0]):
        return np.full(len(values), 0.5, dtype=float)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and np.isclose(values[order[end]], values[order[cursor]]):
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks / max(1, len(values) - 1)


def cosine_similarity(a, b):
    """Returns a bounded cosine similarity and handles empty vectors safely."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-10:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def resample_sequence(values, slots=SEQUENCE_SLOTS):
    """Resamples a one- or two-dimensional bar sequence to a fixed number of slots."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0:
        return np.zeros((slots, values.shape[1]), dtype=float)
    if len(values) == 1:
        return np.repeat(values, slots, axis=0)

    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, slots)
    return np.column_stack([
        np.interp(target, source, values[:, column])
        for column in range(values.shape[1])
    ])


def mean_by_bar(series, bar_starts, bar_ends):
    """Averages a beat-synchronous scalar series inside each detected bar."""
    series = np.asarray(series, dtype=float)
    values = np.zeros(len(bar_starts), dtype=float)
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        clipped_end = min(int(end), len(series))
        values[i] = float(np.mean(series[int(start):clipped_end])) if clipped_end > start else 0.0
    return values


def mean_matrix_by_bar(matrix, bar_starts, bar_ends):
    """Averages each row of a beat-synchronous feature matrix inside every bar."""
    matrix = np.asarray(matrix, dtype=float)
    values = np.zeros((len(bar_starts), matrix.shape[0]), dtype=float)
    for i, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        clipped_end = min(int(end), matrix.shape[1])
        if clipped_end > start:
            values[i] = np.mean(matrix[:, int(start):clipped_end], axis=1)
    return values


def build_bar_features(feats, bar_starts, bar_ends):
    """Builds ordered harmonic, timbral, rhythmic, bass, and energy features."""
    chroma = mean_matrix_by_bar(feats["chroma_sync"], bar_starts, bar_ends)
    mfcc = mean_matrix_by_bar(feats["mfcc_sync"], bar_starts, bar_ends)
    energy = mean_by_bar(feats["full_rms_sync"], bar_starts, bar_ends)
    onset = mean_by_bar(feats["full_onset_sync"], bar_starts, bar_ends)
    drums = mean_by_bar(
        feats["drums_onset_sync"] + feats["drums_rms_sync"],
        bar_starts,
        bar_ends,
    )
    bass = mean_by_bar(feats["bass_rms_sync"], bar_starts, bar_ends)
    centroid = mean_by_bar(feats["centroid_sync"], bar_starts, bar_ends)

    onset_change = np.abs(np.diff(onset, prepend=onset[0])) if len(onset) else onset
    novelty = tie_aware_percentiles(onset_change)

    return {
        "chroma": chroma,
        "mfcc": mfcc,
        "energy": energy,
        "energy_rank": tie_aware_percentiles(energy),
        "onset": onset,
        "onset_rank": tie_aware_percentiles(onset),
        "novelty": novelty,
        "drums": drums,
        "drums_rank": tie_aware_percentiles(drums),
        "bass": bass,
        "bass_rank": tie_aware_percentiles(bass),
        "centroid": centroid,
        "centroid_rank": tie_aware_percentiles(centroid),
    }


def compute_bar_repetition(chroma):
    """Scores each bar by its strongest harmonic match to a distant bar."""
    if len(chroma) == 0:
        return np.asarray([], dtype=float)
    normalized = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-8)
    similarity = normalized @ normalized.T
    scores = np.zeros(len(chroma), dtype=float)
    for i in range(len(chroma)):
        mask = np.abs(np.arange(len(chroma)) - i) >= MIN_LAG_BARS_FOR_REPETITION
        scores[i] = float(np.max(similarity[i, mask])) if np.any(mask) else 0.0
    return tie_aware_percentiles(scores)


def map_raw_boundaries_to_bars(raw_bounds, bar_starts):
    """Maps coarse beat-level structural boundaries to nearest bar positions."""
    positions = []
    for boundary in raw_bounds:
        if len(bar_starts) == 0:
            break
        index = int(np.argmin(np.abs(bar_starts - boundary)))
        positions.append(index)
    return sorted(set(position for position in positions if 0 < position < len(bar_starts)))


def compute_boundary_scores(bar_features, raw_bar_positions):
    """Fuses multifeature change and coarse segmentation evidence at every downbeat."""
    n_bars = len(bar_features["energy"])
    if n_bars == 0:
        return np.asarray([0.0], dtype=float), {}

    chroma_change = np.zeros(n_bars + 1, dtype=float)
    timbre_change = np.zeros(n_bars + 1, dtype=float)
    energy_change = np.zeros(n_bars + 1, dtype=float)
    rhythm_change = np.zeros(n_bars + 1, dtype=float)

    for boundary in range(1, n_bars):
        chroma_change[boundary] = 1.0 - max(
            0.0,
            cosine_similarity(
                bar_features["chroma"][boundary - 1],
                bar_features["chroma"][boundary],
            ),
        )
        timbre_change[boundary] = np.linalg.norm(
            bar_features["mfcc"][boundary] - bar_features["mfcc"][boundary - 1]
        )
        energy_change[boundary] = abs(
            bar_features["energy_rank"][boundary] - bar_features["energy_rank"][boundary - 1]
        )
        rhythm_change[boundary] = abs(
            bar_features["onset_rank"][boundary] - bar_features["onset_rank"][boundary - 1]
        )

    chroma_rank = tie_aware_percentiles(chroma_change[1:-1])
    timbre_rank = tie_aware_percentiles(timbre_change[1:-1])
    energy_rank = tie_aware_percentiles(energy_change[1:-1])
    rhythm_rank = tie_aware_percentiles(rhythm_change[1:-1])

    scores = np.zeros(n_bars + 1, dtype=float)
    coarse = np.zeros(n_bars + 1, dtype=float)
    for position in raw_bar_positions:
        for offset, weight in ((0, 1.0), (-1, 0.5), (1, 0.5)):
            index = position + offset
            if 0 < index < n_bars:
                coarse[index] = max(coarse[index], weight)

    for boundary in range(1, n_bars):
        i = boundary - 1
        scores[boundary] = (
            0.30 * chroma_rank[i]
            + 0.27 * timbre_rank[i]
            + 0.23 * energy_rank[i]
            + 0.20 * rhythm_rank[i]
            + 0.12 * coarse[boundary]
        )

    scores[1:-1] = np.clip(scores[1:-1] / 1.12, 0.0, 1.0)
    scores[0] = 1.0
    scores[-1] = 1.0
    components = {
        "chroma": chroma_change,
        "timbre": timbre_change,
        "energy": energy_change,
        "rhythm": rhythm_change,
        "coarse": coarse,
    }
    return scores, components


def segment_length_cost(length):
    """Returns a soft duration cost favoring 8/16/32-bar K-pop sections."""
    if length in MAIN_PHRASE_LENGTHS:
        return {8: 0.0, 16: 0.30, 32: 0.85}[length]
    nearest = min(MAIN_PHRASE_LENGTHS, key=lambda target: abs(target - length))
    return 0.25 + 0.75 * abs(length - nearest) / nearest


def select_main_boundaries(boundary_scores, n_bars):
    """Uses dynamic programming to select plausible section boundaries without hard snapping."""
    if n_bars <= 0:
        return [0]
    if n_bars < 4:
        return [0, n_bars]

    costs = np.full(n_bars + 1, np.inf, dtype=float)
    previous = np.full(n_bars + 1, -1, dtype=int)
    costs[0] = 0.0

    for end in range(1, n_bars + 1):
        for length in ALLOWED_SEGMENT_LENGTHS:
            start = end - length
            if start < 0 or not np.isfinite(costs[start]):
                continue
            internal = boundary_scores[start + 1:end]
            if length == 32 and len(internal) and np.max(internal) >= 0.58:
                continue
            if length == 16 and len(internal) and np.max(internal) >= 0.78:
                continue
            missed_boundary_cost = 0.35 * float(np.max(internal)) if len(internal) else 0.0
            end_reward = 0.95 * boundary_scores[end] if end < n_bars else 0.65
            cost = costs[start] + segment_length_cost(length) + missed_boundary_cost - end_reward
            if cost < costs[end]:
                costs[end] = cost
                previous[end] = start

        if previous[end] == -1 and end == n_bars:
            for start in range(max(0, end - 36), end):
                if not np.isfinite(costs[start]):
                    continue
                length = end - start
                cost = costs[start] + 1.0 + segment_length_cost(length)
                if cost < costs[end]:
                    costs[end] = cost
                    previous[end] = start

    if previous[n_bars] == -1:
        return [0, n_bars]

    boundaries = [n_bars]
    cursor = n_bars
    while cursor > 0:
        cursor = int(previous[cursor])
        if cursor < 0:
            return [0, n_bars]
        boundaries.append(cursor)
    return sorted(set(boundaries))


def edge_candidate_score(start, end, edge, bar_features, boundary_scores, pickup_seconds=0.0):
    """Scores an intro/outro candidate using its cut, contrast, trend, and conventional length."""
    n_bars = len(bar_features["energy_rank"])
    if not (0 <= start < end <= n_bars):
        return 0.0

    segment_energy = bar_features["energy_rank"][start:end]
    if edge == "intro":
        neighbor = bar_features["energy_rank"][end:min(n_bars, end + 4)]
        cut = end
        trend = float(segment_energy[-1] - segment_energy[0]) if len(segment_energy) > 1 else 0.0
    else:
        neighbor = bar_features["energy_rank"][max(0, start - 4):start]
        cut = start
        trend = float(segment_energy[0] - segment_energy[-1]) if len(segment_energy) > 1 else 0.0

    energy_contrast = abs(float(np.mean(segment_energy)) - float(np.mean(neighbor))) if len(neighbor) else 0.0
    length = end - start
    internal_boundaries = boundary_scores[start + 1:end]
    internal_peak = float(np.max(internal_boundaries)) if len(internal_boundaries) else 0.0
    internal_mean = float(np.mean(internal_boundaries)) if len(internal_boundaries) else 0.0
    if length == 32 and internal_peak >= 0.52:
        return 0.0
    if length == 16 and internal_peak >= 0.72:
        return 0.0

    length_prior = {
        8: 1.0,
        16: 0.82,
        32: 0.55,
        4: 0.76,
    }.get(length, 0.52)
    pickup_bonus = min(1.0, pickup_seconds / 2.0) if edge == "intro" else 0.0
    score = (
        0.44 * boundary_scores[cut]
        + 0.22 * np.clip(energy_contrast, 0.0, 1.0)
        + 0.16 * np.clip(trend, 0.0, 1.0)
        + 0.12 * length_prior
        + 0.06 * pickup_bonus
        - 0.24 * internal_peak
        - 0.10 * internal_mean
    )
    return float(np.clip(score, 0.0, 1.0))


def detect_edge_sections(bar_features, boundary_scores, pickup_seconds):
    """Selects supported intro/outro spans without forcing either role to exist."""
    n_bars = len(bar_features["energy"])
    result = {"intro": None, "outro": None}
    for edge in ("intro", "outro"):
        candidates = []
        for length in EDGE_LENGTHS:
            if length >= n_bars - 3:
                continue
            start, end = (0, length) if edge == "intro" else (n_bars - length, n_bars)
            score = edge_candidate_score(
                start,
                end,
                edge,
                bar_features,
                boundary_scores,
                pickup_seconds,
            )
            cut = end if edge == "intro" else start
            candidates.append((score, boundary_scores[cut], start, end))

        if not candidates:
            continue
        score, cut_strength, start, end = max(candidates)
        if score >= EDGE_SCORE_THRESHOLD and cut_strength >= 0.38:
            result[edge] = {
                "start_bar": start,
                "end_bar": end,
                "score": float(score),
                "pickup_seconds": float(pickup_seconds if edge == "intro" else 0.0),
            }
    return result


def transition_window_score(start, end, bar_features, boundary_scores, repetition):
    """Scores a short transition and validates the stability of its downstream landing."""
    n_bars = len(bar_features["energy"])
    if start <= 0 or end >= n_bars or not (1 <= end - start <= 4):
        return 0.0, 0.0

    novelty = float(np.mean(bar_features["novelty"][start:end]))
    non_repetition = 1.0 - float(np.mean(repetition[start:end]))
    entry_exit = 0.5 * (boundary_scores[start] + boundary_scores[end])
    energy = bar_features["energy_rank"][start:end]
    energy_motion = abs(float(energy[-1] - energy[0])) if len(energy) > 1 else boundary_scores[end]

    downstream_end = min(n_bars, end + 4)
    downstream_changes = boundary_scores[end + 1:downstream_end]
    downstream_stability = 1.0 - float(np.mean(downstream_changes)) if len(downstream_changes) else 0.5
    landing_contrast = boundary_scores[end]
    validation = 0.55 * landing_contrast + 0.45 * downstream_stability

    length_prior = {1: 0.88, 2: 1.0, 3: 0.96, 4: 0.90}[end - start]
    score = length_prior * (
        0.25 * novelty
        + 0.20 * non_repetition
        + 0.20 * entry_exit
        + 0.12 * np.clip(energy_motion, 0.0, 1.0)
        + 0.23 * validation
    )
    return float(np.clip(score, 0.0, 1.0)), float(np.clip(validation, 0.0, 1.0))


def detect_validated_transitions(main_boundaries, edge_sections, bar_features, boundary_scores):
    """Finds nonoverlapping 1-4 bar transitions around proposed section boundaries."""
    repetition = compute_bar_repetition(bar_features["chroma"])
    intro_end = edge_sections["intro"]["end_bar"] if edge_sections["intro"] else 0
    outro_start = edge_sections["outro"]["start_bar"] if edge_sections["outro"] else len(repetition)
    candidates = []

    internal_scores = boundary_scores[1:-1]
    strong_cutoff = max(
        0.55,
        float(np.percentile(internal_scores, 75)) if len(internal_scores) else 1.0,
    )
    strong_downbeats = {
        index for index in range(1, len(boundary_scores) - 1)
        if boundary_scores[index] >= strong_cutoff
    }
    anchors = sorted(set(main_boundaries[1:-1]) | strong_downbeats)

    for boundary in anchors:
        if boundary <= intro_end or boundary >= outro_start:
            continue
        for length in TRANSITION_LENGTHS:
            for start, end in ((boundary - length, boundary), (boundary, boundary + length)):
                if start < intro_end or end > outro_start:
                    continue
                score, validation = transition_window_score(
                    start,
                    end,
                    bar_features,
                    boundary_scores,
                    repetition,
                )
                previous_main = max((value for value in main_boundaries if value <= start), default=0)
                next_main = min(
                    (value for value in main_boundaries if value >= end),
                    default=len(repetition),
                )
                left_remainder = start - previous_main
                right_remainder = next_main - end
                if 0 < left_remainder < 4 or 0 < right_remainder < 4:
                    continue
                if score >= TRANSITION_SCORE_THRESHOLD and validation >= 0.50:
                    candidates.append({
                        "start_bar": int(start),
                        "end_bar": int(end),
                        "bar_count": int(end - start),
                        "score": score,
                        "downstream_validation": validation,
                        "anchor_boundary": int(boundary),
                    })

    selected = []
    occupied = set()
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        bars = set(range(candidate["start_bar"], candidate["end_bar"]))
        if bars & occupied:
            continue
        selected.append(candidate)
        occupied |= bars

    selected.sort(key=lambda item: item["start_bar"])
    consolidated = []
    for candidate in selected:
        if not consolidated or consolidated[-1]["end_bar"] != candidate["start_bar"]:
            consolidated.append(candidate)
            continue

        previous = consolidated[-1]
        combined_length = candidate["end_bar"] - previous["start_bar"]
        if combined_length <= max(TRANSITION_LENGTHS):
            score, validation = transition_window_score(
                previous["start_bar"],
                candidate["end_bar"],
                bar_features,
                boundary_scores,
                repetition,
            )
            if score >= TRANSITION_SCORE_THRESHOLD * 0.90 and validation >= 0.50:
                consolidated[-1] = {
                    "start_bar": previous["start_bar"],
                    "end_bar": candidate["end_bar"],
                    "bar_count": combined_length,
                    "score": score,
                    "downstream_validation": validation,
                    "anchor_boundary": candidate["anchor_boundary"],
                }
                continue

        consolidated[-1] = max((previous, candidate), key=lambda item: item["score"])
    return consolidated


def phrase_grid_confidence(bar_count, is_transition=False):
    """Scores phrase duration against transition or conventional section lengths."""
    targets = TRANSITION_LENGTHS if is_transition else MAIN_PHRASE_LENGTHS
    distance = min(abs(bar_count - target) / target for target in targets)
    confidence = 1.0 - min(1.0, distance)
    if not is_transition and bar_count == 4:
        confidence = max(confidence, 0.65)
    return float(confidence)


def build_phrases(edges, transitions, edge_sections, bar_starts, bar_ends,
                  beat_times, song_duration, boundary_scores, beatgrid_confidence):
    """Builds contiguous phrase records that cover the pickup through the song end."""
    transition_ranges = {
        (item["start_bar"], item["end_bar"]): item
        for item in transitions
    }
    intro_range = None
    outro_range = None
    if edge_sections["intro"]:
        intro_range = (0, edge_sections["intro"]["end_bar"])
    if edge_sections["outro"]:
        outro_range = (edge_sections["outro"]["start_bar"], len(bar_starts))

    phrases = []
    for start_bar, end_bar in zip(edges[:-1], edges[1:]):
        if end_bar <= start_bar:
            continue
        start_beat = int(bar_starts[start_bar])
        end_beat = int(bar_starts[end_bar]) if end_bar < len(bar_starts) else int(bar_ends[-1])
        phrase_range = (start_bar, end_bar)
        transition = transition_ranges.get(phrase_range)
        fixed_role = None
        fixed_score = None
        if phrase_range == intro_range:
            fixed_role = "intro"
            fixed_score = edge_sections["intro"]["score"]
        elif phrase_range == outro_range:
            fixed_role = "outro"
            fixed_score = edge_sections["outro"]["score"]
        elif transition:
            fixed_role = "transition"
            fixed_score = transition["score"]

        start_boundary = beatgrid_confidence if start_bar == 0 else boundary_scores[start_bar]
        end_boundary = beatgrid_confidence if end_bar == len(bar_starts) else boundary_scores[end_bar]
        phrases.append({
            "start_bar": int(start_bar),
            "end_bar": int(end_bar),
            "bar_count": int(end_bar - start_bar),
            "start_beat_index": start_beat,
            "end_beat_index": end_beat,
            "start_time": 0.0 if start_bar == 0 else float(beat_times[start_beat]),
            "end_time": song_duration if end_bar == len(bar_starts) else float(beat_times[end_beat]),
            "pickup_seconds": float(beat_times[0]) if start_bar == 0 else 0.0,
            "pickup_included": bool(start_bar == 0 and beat_times[0] > 0.05),
            "phrase_grid_confidence": phrase_grid_confidence(end_bar - start_bar, transition is not None),
            "start_boundary_confidence": float(np.clip(start_boundary, 0.0, 1.0)),
            "end_boundary_confidence": float(np.clip(end_boundary, 0.0, 1.0)),
            "fixed_role": fixed_role,
            "fixed_role_score": float(fixed_score) if fixed_score is not None else None,
            "transition_validation_score": (
                float(transition["downstream_validation"]) if transition else None
            ),
        })
    return phrases


def add_phrase_descriptors(phrases, bar_features):
    """Adds original-mix energy, rhythm, bass, novelty, and position descriptors."""
    raw_energy = []
    for phrase in phrases:
        start, end = phrase["start_bar"], phrase["end_bar"]
        raw_energy.append(float(np.mean(bar_features["energy"][start:end])))
    energy_levels = tie_aware_percentiles(raw_energy)

    for index, (phrase, energy_level) in enumerate(zip(phrases, energy_levels)):
        start, end = phrase["start_bar"], phrase["end_bar"]
        energy_series = bar_features["energy_rank"][start:end]
        phrase["energy_level"] = float(energy_level)
        phrase["energy_percentile"] = float(energy_level)
        phrase["energy_label"] = ENERGY_LABELS[min(4, int(energy_level * 5))]
        phrase["energy_trend"] = (
            float(energy_series[-1] - energy_series[0]) if len(energy_series) > 1 else 0.0
        )
        phrase["drums_level"] = float(np.mean(bar_features["drums_rank"][start:end]))
        phrase["bass_level"] = float(np.mean(bar_features["bass_rank"][start:end]))
        phrase["novelty_level"] = float(np.mean(bar_features["novelty"][start:end]))
        phrase["position"] = (index + 0.5) / max(1, len(phrases))
    return phrases


def build_phrase_embedding(phrase, bar_features):
    """Builds an ordered phrase representation without collapsing chord or energy order."""
    start, end = phrase["start_bar"], phrase["end_bar"]
    chroma = resample_sequence(bar_features["chroma"][start:end])
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-8
    rhythm = resample_sequence(bar_features["onset_rank"][start:end]).ravel()
    energy = resample_sequence(bar_features["energy_rank"][start:end]).ravel()
    timbre = resample_sequence(bar_features["mfcc"][start:end])
    timbre = (timbre - np.mean(timbre, axis=0, keepdims=True)) / (
        np.std(timbre, axis=0, keepdims=True) + 1e-8
    )
    return {
        "chroma": chroma.ravel(),
        "rhythm": rhythm,
        "energy": energy,
        "timbre": timbre.ravel(),
        "bar_count": phrase["bar_count"],
        "energy_mean": phrase["energy_level"],
    }


def phrase_similarity(a, b):
    """Compares ordered phrase content with explicit energy-level similarity."""
    harmonic = max(0.0, cosine_similarity(a["chroma"], b["chroma"]))
    timbre = 0.5 + 0.5 * cosine_similarity(a["timbre"], b["timbre"])
    rhythm = 0.5 + 0.5 * cosine_similarity(a["rhythm"], b["rhythm"])
    energy_profile = 0.5 + 0.5 * cosine_similarity(a["energy"], b["energy"])
    energy_level = math.exp(-abs(a["energy_mean"] - b["energy_mean"]) / 0.22)
    energy = 0.55 * energy_profile + 0.45 * energy_level
    duration = math.exp(-abs(math.log((a["bar_count"] + 1e-8) / (b["bar_count"] + 1e-8))))

    score = (
        0.38 * harmonic
        + 0.20 * timbre
        + 0.14 * rhythm
        + 0.20 * energy
        + 0.08 * duration
    )
    return float(np.clip(score, 0.0, 1.0))


def cluster_phrase_patterns(phrases, bar_features, threshold=PATTERN_SIMILARITY_THRESHOLD):
    """Complete-link clusters phrases so one weak similarity cannot collapse distinct sections."""
    embeddings = [build_phrase_embedding(phrase, bar_features) for phrase in phrases]
    n_phrases = len(phrases)
    similarity = np.eye(n_phrases, dtype=float)
    for i in range(n_phrases):
        for j in range(i + 1, n_phrases):
            value = phrase_similarity(embeddings[i], embeddings[j])
            similarity[i, j] = similarity[j, i] = value

    clusters = [{index} for index in range(n_phrases)]
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cross = [similarity[a, b] for a in clusters[i] for b in clusters[j]]
                complete_link = min(cross)
                if complete_link >= threshold and (best is None or complete_link > best[0]):
                    best = (complete_link, i, j)
        if best is None:
            break
        _, i, j = best
        clusters[i] |= clusters[j]
        del clusters[j]

    clusters.sort(key=lambda members: min(members))
    for group_id, members in enumerate(clusters):
        members = sorted(members)
        for index in members:
            peer_scores = [similarity[index, peer] for peer in members if peer != index]
            phrases[index]["pattern_group"] = group_id
            phrases[index]["pattern_occurrences"] = len(members)
            phrases[index]["pattern_confidence"] = (
                float(np.mean(peer_scores)) if peer_scores else 0.25
            )
    return similarity


def chorus_group_scores(phrases):
    """Scores repeated groups as chorus candidates without forcing any group to qualify."""
    groups = {}
    for index, phrase in enumerate(phrases):
        if phrase["fixed_role"] in {"intro", "outro", "transition"}:
            continue
        groups.setdefault(phrase["pattern_group"], []).append(index)

    scores = {}
    for group_id, indices in groups.items():
        occurrence_score = min(1.0, (len(indices) - 1) / 2.0)
        energy = float(np.mean([phrases[i]["energy_level"] for i in indices]))
        pattern = float(np.mean([phrases[i]["pattern_confidence"] for i in indices]))
        score = 0.38 * occurrence_score + 0.36 * energy + 0.26 * pattern
        scores[group_id] = float(score if len(indices) >= 2 else score * 0.45)
    return scores


def compute_role_emissions(phrases):
    """Computes role evidence from energy, repetition, position, context, and fixed detections."""
    chorus_scores = chorus_group_scores(phrases)
    repeated_scores = sorted(
        (score for group, score in chorus_scores.items()
         if sum(p["pattern_group"] == group for p in phrases) >= 2),
        reverse=True,
    )
    chorus_cutoff = max(0.58, repeated_scores[1] if len(repeated_scores) > 1 else 0.58)

    emissions = []
    for index, phrase in enumerate(phrases):
        if phrase["fixed_role"]:
            fixed_score = max(0.70, phrase["fixed_role_score"] or 0.70)
            scores = {role: 0.01 for role in CORE_ROLES}
            scores[phrase["fixed_role"]] = fixed_score
            emissions.append(scores)
            continue

        energy = phrase["energy_level"]
        drums = phrase["drums_level"]
        novelty = phrase["novelty_level"]
        position = phrase["position"]
        occurrences = phrase["pattern_occurrences"]
        group_score = chorus_scores.get(phrase["pattern_group"], 0.0)
        chorus_eligible = occurrences >= 2 and group_score >= chorus_cutoff
        next_energy = phrases[index + 1]["energy_level"] if index + 1 < len(phrases) else energy
        previous_energy = phrases[index - 1]["energy_level"] if index > 0 else energy
        rise_to_next = float(np.clip(next_energy - energy + 0.5 * phrase["energy_trend"], 0.0, 1.0))
        fall_from_previous = float(np.clip(previous_energy - energy, 0.0, 1.0))
        late_middle = float(np.clip(1.0 - abs(position - 0.70) / 0.30, 0.0, 1.0))
        uniqueness = 1.0 if occurrences == 1 else 0.25
        conventional_short_section = phrase["bar_count"] in {4, 8}
        pickup_support = min(1.0, phrase["pickup_seconds"] / 2.0)

        chorus = (
            0.38 * group_score
            + 0.30 * energy
            + 0.18 * min(1.0, occurrences / 3.0)
            + 0.14 * phrase["pattern_confidence"]
        )
        if not chorus_eligible:
            chorus *= 0.48
        final_chorus_context = (
            position >= 0.72
            and energy >= 0.60
            and index > 0
            and phrases[index - 1]["fixed_role"] == "transition"
        )
        if final_chorus_context:
            chorus += 0.22

        scores = {
            "intro": (
                0.12
                + 0.34 * (index == 0)
                + 0.16 * conventional_short_section
                + 0.10 * pickup_support
                + 0.10 * (1.0 - energy)
            ),
            "verse": 0.28 + 0.30 * (1.0 - abs(energy - 0.38)) + 0.18 * uniqueness + 0.12 * (position < 0.75),
            "pre-chorus": 0.18 + 0.42 * rise_to_next + 0.22 * (0.25 <= energy <= 0.80) + 0.10 * novelty,
            "chorus": chorus,
            "post-chorus": 0.16 + 0.30 * drums + 0.24 * novelty + 0.22 * fall_from_previous,
            "bridge": 0.16 + 0.30 * uniqueness + 0.25 * late_middle + 0.17 * abs(energy - previous_energy),
            "instrumental": 0.12 + 0.30 * drums + 0.22 * novelty + 0.18 * uniqueness + 0.10 * energy,
            "transition": 0.08 + 0.30 * novelty + 0.18 * abs(phrase["energy_trend"]),
            "outro": (
                0.10
                + 0.38 * (index == len(phrases) - 1)
                + 0.14 * (phrase["bar_count"] <= 8)
                + 0.18 * (1.0 - energy)
                + 0.10 * np.clip(-phrase["energy_trend"], 0.0, 1.0)
            ),
        }
        emissions.append({role: float(np.clip(value, 0.01, 0.99)) for role, value in scores.items()})
    return emissions


def transition_prior(previous, current):
    """Returns a soft K-pop form prior while leaving nontraditional sequences possible."""
    preferred = {
        "intro": {"verse": 0.78, "chorus": 0.40, "instrumental": 0.44},
        "verse": {"pre-chorus": 0.82, "chorus": 0.45, "verse": 0.30, "transition": 0.42},
        "pre-chorus": {"chorus": 0.92, "transition": 0.46, "pre-chorus": 0.12},
        "chorus": {"verse": 0.66, "post-chorus": 0.74, "bridge": 0.52, "outro": 0.38, "transition": 0.48},
        "post-chorus": {"verse": 0.70, "bridge": 0.44, "transition": 0.48, "outro": 0.34},
        "bridge": {"chorus": 0.88, "outro": 0.38, "transition": 0.48},
        "instrumental": {"verse": 0.52, "chorus": 0.48, "transition": 0.50, "outro": 0.36},
        "transition": {"verse": 0.52, "pre-chorus": 0.55, "chorus": 0.72, "bridge": 0.44, "instrumental": 0.42},
        "outro": {"outro": 0.60},
    }
    if current in preferred.get(previous, {}):
        return preferred[previous][current]
    if previous == current:
        return 0.10 if current == "chorus" else 0.22
    return 0.16


def decode_roles(phrases, emissions):
    """Viterbi-decodes phrase roles using emissions and soft K-pop sequence priors."""
    if not phrases:
        return []
    n_roles = len(CORE_ROLES)
    n_phrases = len(phrases)
    scores = np.full((n_phrases, n_roles), -np.inf, dtype=float)
    back = np.full((n_phrases, n_roles), -1, dtype=int)
    start_prior = {
        "intro": 0.52,
        "verse": 0.46,
        "chorus": 0.20,
        "instrumental": 0.28,
        "transition": 0.08,
        "outro": 0.02,
    }

    for role_index, role in enumerate(CORE_ROLES):
        scores[0, role_index] = math.log(emissions[0][role]) + math.log(start_prior.get(role, 0.12))

    for phrase_index in range(1, n_phrases):
        for current_index, current in enumerate(CORE_ROLES):
            emission = math.log(emissions[phrase_index][current])
            candidates = [
                scores[phrase_index - 1, previous_index]
                + math.log(transition_prior(previous, current))
                + emission
                for previous_index, previous in enumerate(CORE_ROLES)
            ]
            best_previous = int(np.argmax(candidates))
            scores[phrase_index, current_index] = candidates[best_previous]
            back[phrase_index, current_index] = best_previous

    end_bonus = np.asarray([
        math.log(0.52 if role == "outro" else 0.28 if role in {"chorus", "post-chorus"} else 0.20)
        for role in CORE_ROLES
    ])
    current = int(np.argmax(scores[-1] + end_bonus))
    path = [current]
    for phrase_index in range(n_phrases - 1, 0, -1):
        current = int(back[phrase_index, current])
        path.append(current)
    path.reverse()
    return [CORE_ROLES[index] for index in path]


def refine_decoded_roles(phrases, roles, emissions):
    """Rejects unsupported pre-choruses and resolves adjacent chorus saturation."""
    roles = list(roles)
    for index, role in enumerate(roles):
        if role == "pre-chorus":
            next_index = index + 1
            while next_index < len(roles) and roles[next_index] == "transition":
                next_index += 1
            next_core = roles[next_index] if next_index < len(roles) else None
            if next_core not in {"chorus", "transition"}:
                roles[index] = "bridge" if phrases[index]["position"] > 0.55 else "verse"

    for index in range(1, len(roles)):
        if roles[index - 1] == roles[index] == "chorus":
            previous_score = emissions[index - 1]["chorus"]
            current_score = emissions[index]["chorus"]
            relabel_index = index - 1 if previous_score < current_score else index
            alternative_roles = ("post-chorus", "verse", "instrumental", "bridge")
            roles[relabel_index] = max(
                alternative_roles,
                key=lambda candidate: emissions[relabel_index][candidate],
            )
    return roles


def add_role_confidence(phrases, roles, emissions):
    """Adds role confidence from local evidence, ambiguity margin, and sequence support."""
    for index, (phrase, role, role_scores) in enumerate(zip(phrases, roles, emissions)):
        ordered = sorted(role_scores.values(), reverse=True)
        chosen = role_scores[role]
        second = max((score for candidate, score in role_scores.items() if candidate != role), default=0.0)
        margin = float(np.clip((chosen - second + 0.5) / 1.5, 0.0, 1.0))
        if index == 0:
            sequence_support = 0.70
        else:
            sequence_support = transition_prior(roles[index - 1], role)
        role_confidence = 0.58 * chosen + 0.24 * margin + 0.18 * sequence_support
        if phrase["fixed_role"] == role:
            role_confidence = 0.70 * (phrase["fixed_role_score"] or 0.70) + 0.30 * role_confidence

        phrase["phrase_role"] = role
        phrase["role_confidence"] = float(np.clip(role_confidence, 0.0, 1.0))
        phrase["role_score_margin"] = margin
        phrase["role_scores"] = {key: float(value) for key, value in role_scores.items()}
    return phrases


def add_phrase_confidences(phrases, beatgrid_confidence):
    """Combines segmentation and identification evidence conservatively for every phrase."""
    for phrase in phrases:
        boundary_confidence = math.sqrt(
            max(0.0, phrase["start_boundary_confidence"])
            * max(0.0, phrase["end_boundary_confidence"])
        )
        structural = 0.62 * boundary_confidence + 0.38 * phrase["phrase_grid_confidence"]
        pattern = phrase["pattern_confidence"] if phrase["pattern_occurrences"] > 1 else 0.50
        identification = 0.78 * phrase["role_confidence"] + 0.22 * pattern
        confidence = (
            0.48 * structural
            + 0.42 * identification
            + 0.10 * beatgrid_confidence
        )
        phrase["phrase_confidence_components"] = {
            "structural": float(np.clip(structural, 0.0, 1.0)),
            "role": phrase["role_confidence"],
            "pattern": float(pattern),
            "beatgrid": float(beatgrid_confidence),
        }
        phrase["phrase_confidence"] = float(np.clip(confidence, 0.0, 1.0))
    return phrases


def overall_phrase_confidence(phrases):
    """Returns a bar-duration-weighted confidence score for the full phrase timeline."""
    if not phrases:
        return 0.0
    weights = np.asarray([phrase["bar_count"] for phrase in phrases], dtype=float)
    values = np.asarray([phrase["phrase_confidence"] for phrase in phrases], dtype=float)
    return float(np.average(values, weights=weights))


def validate_phrase_timeline(phrases, n_bars, song_duration):
    """Ensures phrase ranges are contiguous, nonoverlapping, and cover the complete song."""
    if not phrases:
        raise ValueError("Phrase analysis produced no phrases")
    if phrases[0]["start_bar"] != 0 or phrases[-1]["end_bar"] != n_bars:
        raise ValueError("Phrase timeline does not cover all bars")
    for previous, current in zip(phrases[:-1], phrases[1:]):
        if previous["end_bar"] != current["start_bar"]:
            raise ValueError("Phrase timeline contains a gap or overlap")
    phrases[0]["start_time"] = 0.0
    phrases[-1]["end_time"] = float(song_duration)


def incomplete_result(beatgrid_confidence, beats=None, downbeats=None):
    """Builds a consistent low-confidence result when phrase analysis cannot proceed."""
    return {
        "status": "incomplete",
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "model_version": ANALYSIS_MODEL_VERSION,
        "analysis_source": ANALYSIS_SOURCE,
        "tempo": 0.0,
        "beatgrid_confidence": float(beatgrid_confidence),
        "song_key": "N",
        "song_scale": "none",
        "song_key_strength": 0.0,
        "overall_phrase_confidence": 0.0,
        "phrases": [],
        "beats": list(beats or []),
        "downbeats": list(downbeats or []),
    }


def analyze_song(wav_path: Path):
    """Runs the original-mix segmentation, role, confidence, key, and energy pipeline."""
    y_full, sr = load_mono(wav_path)
    song_duration = float(len(y_full) / sr)

    beat_times, bar_index, _, beatgrid_confidence = madmom_downbeats(y_full, sr)
    if beat_times.size == 0:
        return incomplete_result(beatgrid_confidence)

    beat_times, bar_index = extend_beatgrid_to_duration(
        beat_times,
        bar_index,
        song_duration,
    )
    downbeat_times = beat_times[bar_index == 1].tolist()
    feats = build_mix_beat_sync_features(y_full, sr, beat_times)
    if feats is None:
        return incomplete_result(
            beatgrid_confidence,
            beats=beat_times.tolist(),
            downbeats=downbeat_times,
        )

    affinity, n_beats = fused_affinity_matrix(feats)
    bar_vectors, bar_starts, bar_ends = compute_bar_level_features(feats, downbeat_times)
    if bar_vectors is None or len(bar_vectors) < 4:
        return incomplete_result(
            beatgrid_confidence,
            beats=feats["beat_times"].tolist(),
            downbeats=downbeat_times,
        )

    bar_features = build_bar_features(feats, bar_starts, bar_ends)
    raw_bounds = laplacian_segment(affinity, n_beats)
    raw_bar_positions = map_raw_boundaries_to_bars(raw_bounds, bar_starts)
    boundary_scores, _ = compute_boundary_scores(bar_features, raw_bar_positions)
    main_boundaries = select_main_boundaries(boundary_scores, len(bar_starts))

    pickup_seconds = float(feats["beat_times"][0]) if len(feats["beat_times"]) else 0.0
    edge_sections = detect_edge_sections(bar_features, boundary_scores, pickup_seconds)
    transitions = detect_validated_transitions(
        main_boundaries,
        edge_sections,
        bar_features,
        boundary_scores,
    )

    edges = set(main_boundaries)
    for edge in edge_sections.values():
        if edge:
            edges.update((edge["start_bar"], edge["end_bar"]))
    for transition in transitions:
        edges.update((transition["start_bar"], transition["end_bar"]))
    protected_spans = [
        (edge["start_bar"], edge["end_bar"])
        for edge in edge_sections.values()
        if edge
    ] + [
        (transition["start_bar"], transition["end_bar"])
        for transition in transitions
    ]
    edges = {
        edge for edge in edges
        if not any(start < edge < end for start, end in protected_spans)
    }
    edges = sorted(edge for edge in edges if 0 <= edge <= len(bar_starts))

    phrases = build_phrases(
        edges,
        transitions,
        edge_sections,
        bar_starts,
        bar_ends,
        feats["beat_times"],
        song_duration,
        boundary_scores,
        beatgrid_confidence,
    )
    add_phrase_descriptors(phrases, bar_features)
    cluster_phrase_patterns(phrases, bar_features)
    emissions = compute_role_emissions(phrases)
    roles = decode_roles(phrases, emissions)
    roles = refine_decoded_roles(phrases, roles, emissions)
    add_role_confidence(phrases, roles, emissions)
    add_phrase_confidences(phrases, beatgrid_confidence)
    validate_phrase_timeline(phrases, len(bar_starts), song_duration)

    tempo = float(60.0 / (np.mean(np.diff(beat_times)) + 1e-8)) if len(beat_times) >= 2 else 0.0
    key_pipeline = build_key_extractor()
    song_key, song_scale, song_key_strength = estimate_key_for_slice(
        y_full,
        sr,
        key_pipeline,
    )
    for phrase in phrases:
        start_sample = int(phrase["start_time"] * sr)
        end_sample = int(phrase["end_time"] * sr)
        key, scale, strength = estimate_key_for_slice(
            y_full[start_sample:end_sample],
            sr,
            key_pipeline,
        )
        phrase["key"] = key
        phrase["scale"] = scale
        phrase["key_strength"] = strength
        phrase["key_matches_song"] = bool(key == song_key and scale == song_scale)

    energy_result = compute_energy_score(y_full, sr, feats, tempo)
    return {
        "status": "complete",
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "model_version": ANALYSIS_MODEL_VERSION,
        "analysis_source": ANALYSIS_SOURCE,
        "tempo": tempo,
        "beatgrid_confidence": float(beatgrid_confidence),
        "song_key": song_key,
        "song_scale": song_scale,
        "song_key_strength": song_key_strength,
        "overall_phrase_confidence": overall_phrase_confidence(phrases),
        "energy": energy_result["energy"],
        "energy_components": energy_result["energy_components"],
        "phrases": phrases,
        "validated_transitions": transitions,
        "edge_sections": edge_sections,
        "beats": feats["beat_times"].tolist(),
        "downbeats": [float(feats["beat_times"][index]) for index in bar_starts],
    }


def save_analysis(song_wav: Path, result: dict):
    """Writes the canonical analysis output beside the song audio."""
    output_path = song_wav.parent / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return output_path


def analyze_and_save(song_wav: Path):
    """Analyzes one WAV, saves it, and prints a concise structural summary."""
    print(f"[analysis-v3:original-mix] {song_wav}")
    result = analyze_song(song_wav)
    output_path = save_analysis(song_wav, result)
    role_counts = {
        role: sum(phrase["phrase_role"] == role for phrase in result["phrases"])
        for role in CORE_ROLES
    }
    active_counts = ", ".join(
        f"{role}={count}" for role, count in role_counts.items() if count
    )
    print(
        f"[done] tempo={result['tempo']:.1f} | "
        f"phrase_conf={result['overall_phrase_confidence']:.2f} | "
        f"phrases={len(result['phrases'])} | {active_counts} -> {output_path}"
    )
    return result


def main(argv=None):
    """Analyzes one requested song or every song when no argument is supplied."""
    parser = argparse.ArgumentParser(
        description="Sequence-aware K-pop phrase analysis",
    )
    parser.add_argument(
        "--song",
        type=str,
        default=None,
        help="Song directory name or partial name; omit to analyze all songs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Persistent song-library directory (default: project data directory)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip songs that already contain analysis.json",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if args.song:
        song_wav = find_song_wav(args.song, data_dir)
        if args.skip_existing and has_current_analysis(song_wav.parent):
            print(f"[skip:existing] {song_wav.parent.name}")
            return
        analyze_and_save(song_wav)
        return

    wavs = list_song_wavs(data_dir)
    if args.skip_existing:
        wavs = [wav for wav in wavs if not has_current_analysis(wav.parent)]
    if not wavs:
        print(f"No WAV files require analysis in {data_dir}")
        return
    failures = []
    for wav in wavs:
        try:
            analyze_and_save(wav)
        except Exception as exc:
            print(f"[error] {wav}: {exc}")
            failures.append((wav, exc))
    if failures:
        raise RuntimeError(f"Analysis failed for {len(failures)} song(s)")


if __name__ == "__main__":
    main()
