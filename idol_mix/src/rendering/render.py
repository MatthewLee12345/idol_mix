import argparse
import json
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

try:
    import pyrubberband as pyrb
    _HAVE_RUBBERBAND = True
except Exception:
    _HAVE_RUBBERBAND = False


SR = 44100  # bumped from 22050 -> full audible bandwidth (~22kHz) instead of ~11kHz
VOCAL_ANALYSIS_SR = 22050
VOCAL_ANALYSIS_HOP_LENGTH = 512
VOCAL_ACTIVE_PEAK_FRACTION = 0.18
VOCAL_PRESENT_PEAK_FRACTION = 0.06
VOCAL_ACTIVE_WINDOW_FRACTION = 0.18
VOCAL_PRESENT_WINDOW_FRACTION = 0.05
BEATS_PER_BAR = 4
MAX_BPM_STRETCH_DELTA = 5.0

FORBIDDEN_ROLES = {"intro", "outro", "transition"}
MIN_RENDER_BARS = 8
MIXER_OVERLAP_TARGET_BARS = 8
MIXER_MAX_OVERLAP_BARS = 16
MIXER_MAX_COMBINED_PHRASES = 3
MIXER_COMBINE_MIN_PHRASE_CONFIDENCE = 0.68
LEAD_IN_BARS = 4
TRANSITION_CROSSFADE_BARS = 4
TRANSITION_SOLO_BARS = 8
MIXER_INCOMING_VOCAL_BOOST_DB = 1.0
MIXER_MIN_INCOMING_VOCAL_OCCUPANCY = 0.45
MIXER_MIN_INCOMING_VOCAL_RMS = 0.004
MIXER_MIN_PREROLL_VOCAL_OCCUPANCY = 0.10
MIXER_MIN_PREROLL_VOCAL_RMS = 0.002

EQ_LOW_CUTOFF_HZ = 250.0
EQ_HIGH_CUTOFF_HZ = 4000.0
EQ_BASS_SWAP_FRACTION = 0.5
EQ_FILTER_ORDER = 4           # steeper rolloff than before (was 2) -> cleaner band separation, less bleed

OUTPUT_SUBTYPE = "PCM_24"     # 24-bit output instead of default 16-bit -> more headroom, less quantization noise
INTERNAL_DTYPE = np.float64   # process internally in float64, only cast down at the very final write

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))
DEFAULT_RENDERS_DIR = Path(os.environ.get("KPOP_DJ_RENDERS_DIR", PROJECT_ROOT / "renders"))
COMPATIBILITY_FILENAME = "compatability_list.json"
_COMPATIBILITY_CACHE = {}
_VOCAL_TIMELINE_CACHE = {}


# ---------------------------------------------------------------------------
# Basic I/O helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio(path, sr=SR):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(INTERNAL_DTYPE)


def save_audio(path, y, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    y_out = np.asarray(y, dtype=np.float64)
    sf.write(str(path), y_out, sr, subtype=OUTPUT_SUBTYPE)


def write_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def normalize_peak(y, peak=0.98):
    if len(y) == 0:
        return y
    mx = float(np.max(np.abs(y)))
    if mx <= 1e-8:
        return y
    return y * (peak / mx)


def soft_limit(y, ceiling=0.98, knee=0.9):
    """
    Soft-knee limiter applied before the final hard peak-normalize. Summing
    multiple stems can push transient peaks well above the sustained level;
    a hard normalize alone just scales the whole signal down around that
    one peak, which can make the rest of the mix sound quieter/duller than
    necessary. This gently compresses only the portion of the signal above
    `knee`, preserving more perceived loudness/detail in the body of the mix.
    """
    if not 0.0 < ceiling <= 1.0 or knee <= 0.0:
        raise ValueError("soft limiter requires a positive knee and 0 < ceiling <= 1")
    knee = min(knee, ceiling * 0.9)
    if len(y) == 0:
        return y
    mx = float(np.max(np.abs(y)))
    if mx <= knee:
        return y
    sign = np.sign(y)
    mag = np.abs(y)
    over = mag > knee
    width = ceiling - knee
    compressed = knee + width * np.tanh((mag[over] - knee) / width)
    mag[over] = compressed
    return sign * mag


def db_to_gain(db):
    return float(10.0 ** (db / 20.0))


def incoming_tempo_ratio(outgoing_tempo, incoming_tempo):
    """Validate the +/-5 BPM limit and return the incoming-only stretch ratio."""
    if outgoing_tempo <= 0 or incoming_tempo <= 0:
        raise ValueError("Both track tempos must be positive")
    bpm_delta = abs(outgoing_tempo - incoming_tempo)
    if bpm_delta > MAX_BPM_STRETCH_DELTA + 1e-8:
        raise ValueError(
            f"BPM difference {bpm_delta:.2f} exceeds the allowed "
            f"+/-{MAX_BPM_STRETCH_DELTA:.0f} BPM stretch"
        )
    return outgoing_tempo / incoming_tempo


def rms_of(y):
    if y is None or len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))


def load_vocal_timeline(vocal_path):
    """Load and cache a frame-RMS timeline for one separated vocal stem."""
    path = Path(vocal_path).resolve()
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _VOCAL_TIMELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    for stale_key in [key for key in _VOCAL_TIMELINE_CACHE if key[0] == str(path)]:
        del _VOCAL_TIMELINE_CACHE[stale_key]
    audio = load_audio(path, sr=VOCAL_ANALYSIS_SR)
    frame_rms = librosa.feature.rms(
        y=audio.astype(np.float32),
        hop_length=VOCAL_ANALYSIS_HOP_LENGTH,
    )[0].astype(np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(len(frame_rms)),
        sr=VOCAL_ANALYSIS_SR,
        hop_length=VOCAL_ANALYSIS_HOP_LENGTH,
    ).astype(np.float32)
    peak_rms = float(np.max(frame_rms)) if len(frame_rms) else 0.0
    timeline = (frame_rms, frame_times, peak_rms)
    _VOCAL_TIMELINE_CACHE[cache_key] = timeline
    return timeline


def vocal_window_metrics(frame_rms, frame_times, peak_rms, start_sec, end_sec):
    """Measure normalized vocal occurrence and strength inside one source window."""
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    left = int(np.searchsorted(frame_times, start, side="left"))
    right = int(np.searchsorted(frame_times, end, side="left"))
    left = min(left, len(frame_rms))
    right = min(right, len(frame_rms))
    values = frame_rms[left:right]
    if len(values) == 0 or peak_rms <= 1e-8:
        return {
            "vocal_state": "no_vocals",
            "vocal_level": "none",
            "vocal_occupancy": 0.0,
            "vocal_activity_level": 0.0,
            "vocal_rms": 0.0,
        }

    active_fraction = float(np.mean(values >= peak_rms * VOCAL_ACTIVE_PEAK_FRACTION))
    present_fraction = float(np.mean(values >= peak_rms * VOCAL_PRESENT_PEAK_FRACTION))
    activity_level = float(np.mean(np.clip(values / (peak_rms * 0.5), 0.0, 1.0)))
    if active_fraction >= VOCAL_ACTIVE_WINDOW_FRACTION:
        state, level, occupancy = "active_vocals", "high", active_fraction
    elif present_fraction >= VOCAL_PRESENT_WINDOW_FRACTION:
        state, level, occupancy = "sparse_vocals", "low", present_fraction
    else:
        state, level, occupancy = "no_vocals", "none", present_fraction
    return {
        "vocal_state": state,
        "vocal_level": level,
        "vocal_occupancy": float(np.clip(occupancy, 0.0, 1.0)),
        "vocal_activity_level": float(np.clip(activity_level, 0.0, 1.0)),
        "vocal_rms": rms_of(values),
    }


def refresh_vocal_occurrence(analysis, vocal_path):
    """Enrich loaded phrase data from vocals.wav without changing analysis.json."""
    phrases = analysis.get("phrases")
    if not isinstance(phrases, list):
        raise ValueError("analysis.json has no phrase list for render-time vocal analysis")
    tempo = float(analysis.get("tempo", 0.0))
    bar_duration = bar_duration_seconds(tempo)
    frame_rms, frame_times, peak_rms = load_vocal_timeline(vocal_path)
    for phrase in phrases:
        start = phrase_start(phrase)
        end = phrase_end(phrase)
        phrase.update(vocal_window_metrics(frame_rms, frame_times, peak_rms, start, end))
        prefix_metrics = {}
        prefix_limit = min(
            MIXER_MAX_OVERLAP_BARS,
            max(1, int(np.ceil(phrase_bars(phrase)))),
        )
        for bars in range(1, prefix_limit + 1):
            prefix_metrics[str(bars)] = vocal_window_metrics(
                frame_rms,
                frame_times,
                peak_rms,
                start,
                min(end, start + bars * bar_duration),
            )
        phrase["vocal_prefix_metrics"] = prefix_metrics
        preroll = vocal_window_metrics(
            frame_rms,
            frame_times,
            peak_rms,
            max(0.0, start - bar_duration),
            start,
        )
        phrase["vocal_preroll_occupancy"] = preroll["vocal_occupancy"]
        phrase["vocal_preroll_rms"] = preroll["vocal_rms"]
    return {
        "source": "render_stage_vocals_stem",
        "algorithm": "frame_rms_peak_fraction_v1",
        "sample_rate": VOCAL_ANALYSIS_SR,
        "hop_length": VOCAL_ANALYSIS_HOP_LENGTH,
        "active_peak_fraction": VOCAL_ACTIVE_PEAK_FRACTION,
        "present_peak_fraction": VOCAL_PRESENT_PEAK_FRACTION,
        "active_window_fraction": VOCAL_ACTIVE_WINDOW_FRACTION,
        "present_window_fraction": VOCAL_PRESENT_WINDOW_FRACTION,
        "peak_rms": peak_rms,
    }

def fade_envelope(n, start, end, fade_in):
    """Build a bounded equal-power fade over an absolute sample range."""
    if n < 0 or not 0 <= start <= end <= n:
        raise ValueError("invalid fade envelope bounds")
    gain = np.ones(n, dtype=np.float64) if not fade_in else np.zeros(n, dtype=np.float64)
    if end == start:
        gain[start:] = 1.0 if fade_in else 0.0
        return gain
    t = np.linspace(0.0, np.pi / 2.0, end - start, dtype=np.float64)
    gain[start:end] = np.sin(t) if fade_in else np.cos(t)
    gain[end:] = 1.0 if fade_in else 0.0
    return gain


def overlapping_handoff_envelopes(n, start, end):
    """Crossfade outgoing and incoming vocals together over one shared range."""
    if not 0 <= start < end <= n:
        raise ValueError("invalid vocal handoff bounds")
    out_gain = fade_envelope(n, start, end, fade_in=False)
    in_gain = fade_envelope(n, start, end, fade_in=True)
    return out_gain, in_gain


def apply_envelope(y, gain):
    """Apply a same-length gain envelope to a signal."""
    if len(y) != len(gain):
        raise ValueError("signal and envelope lengths differ")
    return (np.asarray(y, dtype=np.float64) * gain).astype(INTERNAL_DTYPE)


def time_stretch_audio(y, sr, tempo_ratio):
    """
    High-quality time-stretch. Prefers pyrubberband (wraps the Rubber Band
    library -- industry-standard, far cleaner on vocals/transients than a
    basic phase vocoder) and falls back to librosa's phase vocoder only if
    the rubberband binary/bindings aren't available in this environment.
    """
    if len(y) == 0:
        return y
    if tempo_ratio <= 0:
        raise ValueError("tempo_ratio must be positive")
    if abs(tempo_ratio - 1.0) < 1e-4:
        return y

    if _HAVE_RUBBERBAND:
        try:
            stretched = pyrb.time_stretch(y.astype(np.float64), sr, tempo_ratio)
            return stretched.astype(INTERNAL_DTYPE)
        except Exception as e:
            warnings.warn(
                f"pyrubberband time-stretch failed ({e}); falling back to librosa phase vocoder"
            )

    stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=tempo_ratio)
    return stretched.astype(INTERNAL_DTYPE)

def build_stretched_incoming_stems(incoming_bundle, tempo_ratio, sr=SR):
    """
    Loads every available incoming stem and time-stretches EACH ONE EXACTLY
    ONCE, using the single fixed tempo_ratio for the whole track. All later
    slicing (for the mixer overlap window or the transition crossfade/solo
    windows) operates on these already-stretched arrays -- nothing gets
    re-stretched, and no per-segment or time-varying ratio is ever applied.
    """
    stem_paths = incoming_bundle["stem_paths"]
    stretched = {}
    for stem_name, stem_path in stem_paths.items():
        p = Path(stem_path)
        if not p.is_file():
            continue
        y_raw = load_audio(p, sr=sr)
        stretched[stem_name] = time_stretch_audio(y_raw, sr, tempo_ratio)
    if not stretched:
        raise FileNotFoundError("No usable incoming stems found to time-stretch")
    return stretched


def sum_stem_dict(stretched_stems, exclude=None):
    exclude = exclude or set()
    ys = [y for name, y in stretched_stems.items() if name not in exclude]
    if not ys:
        raise FileNotFoundError("No usable stems found in stretched stem set")
    n = max(len(y) for y in ys)
    mix = np.zeros(n, dtype=INTERNAL_DTYPE)
    for y in ys:
        mix[:len(y)] += y
    return mix


def slice_planned_window(y, sr, start_sec, sample_count, label):
    """Slice an exact planned window, rejecting materially short source audio."""
    if start_sec < 0 or sample_count <= 0:
        raise ValueError(f"Invalid planned window for {label}")
    start = int(round(start_sec * sr))
    end = start + int(sample_count)
    if start >= len(y) or end > len(y):
        available = max(0, len(y) - start)
        raise ValueError(
            f"{label} is too short: planned {sample_count} samples, available {available}"
        )
    return np.asarray(y[start:end], dtype=INTERNAL_DTYPE)


def sum_available_stems(stem_paths, exclude=None, sr=SR):
    exclude = exclude or set()
    ys = []
    for stem_name, stem_path in stem_paths.items():
        if stem_name in exclude:
            continue
        p = Path(stem_path)
        if p.is_file():
            ys.append(load_audio(p, sr=sr))
    if not ys:
        raise FileNotFoundError("No usable stems found")
    n = max(len(y) for y in ys)
    mix = np.zeros(n, dtype=INTERNAL_DTYPE)
    for y in ys:
        mix[:len(y)] += y
    return mix


def load_single_stem(stem_paths, stem_name, sr=SR):
    p = stem_paths.get(stem_name)
    if p is None or not Path(p).is_file():
        return None
    return load_audio(Path(p), sr=sr)


# ---------------------------------------------------------------------------
# EQ-curve crossfade (band-specific, DJ-style "bass swap" + smooth mid/high)
# ---------------------------------------------------------------------------

def band_split(y, sr, low_cutoff=EQ_LOW_CUTOFF_HZ, high_cutoff=EQ_HIGH_CUTOFF_HZ, order=EQ_FILTER_ORDER):
    """
    Splits audio into low / mid / high bands using zero-phase Butterworth
    filters. Order bumped to 4 (from 2) for steeper rolloff -- less bleed
    between bands, so the bass swap and mid/high fades stay cleaner and
    don't muddy each other during the crossfade.
    """
    if sr <= 0 or order < 1 or not 0.0 < low_cutoff < high_cutoff < sr / 2.0:
        raise ValueError("band split requires 0 < low cutoff < high cutoff < Nyquist")
    if len(y) == 0:
        return y, y, y
    y64 = y.astype(np.float64)
    sos_low = butter(order, low_cutoff, btype="low", fs=sr, output="sos")
    sos_high = butter(order, high_cutoff, btype="high", fs=sr, output="sos")
    try:
        low = sosfiltfilt(sos_low, y64)
        high = sosfiltfilt(sos_high, y64)
    except ValueError:
        zeros = np.zeros_like(y64, dtype=INTERNAL_DTYPE)
        return zeros, y64.astype(INTERNAL_DTYPE), zeros
    mid = y64 - low - high
    return low.astype(INTERNAL_DTYPE), mid.astype(INTERNAL_DTYPE), high.astype(INTERNAL_DTYPE)


def equal_power_curve(n):
    """Return complementary equal-power fade-out and fade-in curves."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    return fade_out, fade_in


def eq_curve_crossfade(out_seg, in_seg, sr, bass_swap_fraction=EQ_BASS_SWAP_FRACTION):
    """Crossfade two signals with an early bass swap and full-band EQ curves."""
    if not 0.0 < bass_swap_fraction <= 1.0:
        raise ValueError("bass_swap_fraction must be in (0, 1]")
    n = min(len(out_seg), len(in_seg))
    if n <= 0:
        return np.zeros(0, dtype=INTERNAL_DTYPE)
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    out_low, out_mid, out_high = band_split(out_seg, sr)
    in_low, in_mid, in_high = band_split(in_seg, sr)

    fade_out_mh, fade_in_mh = equal_power_curve(n)

    bass_n = max(1, int(round(n * bass_swap_fraction)))
    fade_out_bass_partial, fade_in_bass_partial = equal_power_curve(bass_n)
    fade_out_low = np.concatenate([fade_out_bass_partial, np.zeros(n - bass_n, dtype=np.float64)])
    fade_in_low = np.concatenate([fade_in_bass_partial, np.ones(n - bass_n, dtype=np.float64)])

    mixed_low = out_low * fade_out_low + in_low * fade_in_low
    mixed_mid = out_mid * fade_out_mh + in_mid * fade_in_mh
    mixed_high = out_high * fade_out_mh + in_high * fade_in_mh

    return (mixed_low + mixed_mid + mixed_high).astype(INTERNAL_DTYPE)


# ---------------------------------------------------------------------------
# Phrase-field accessors (fields come from analysis.json "phrases" entries)
# ---------------------------------------------------------------------------

def phrase_role(p):
    return p.get("phrase_role", "unknown")


def phrase_bars(p):
    value = p.get("bar_count")
    if value is None:
        value = p.get("bar_count_estimate")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return 0.0
    return float(value)


def phrase_start(p):
    return float(p.get("start_time", 0.0))


def phrase_end(p):
    return float(p.get("end_time", 0.0))


def vocal_occ_fraction(p):
    value = p.get("vocal_occupancy", 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def phrase_vocal_prefix_metric(phrase, bars, key):
    """Read a render-time vocal metric over the requested leading bars."""
    if bars >= phrase_bars(phrase) - 1e-6:
        return float(phrase.get(key, 0.0))
    prefix = phrase.get("vocal_prefix_metrics", {})
    metrics = prefix.get(str(max(1, int(round(bars)))), {})
    return float(metrics.get(key, 0.0))


def phrase_role_sequence(p):
    """Return the ordered structural roles represented by a phrase or span."""
    roles = p.get("phrase_role_sequence")
    return tuple(roles) if roles else (phrase_role(p),)


def bar_duration_seconds(tempo_bpm):
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")
    return (60.0 / tempo_bpm) * BEATS_PER_BAR


def start_is_acceptable(p):
    return p.get("start_boundary_confidence", 0.0) >= 0.35


def end_is_acceptable(p):
    return p.get("end_boundary_confidence", 0.0) >= 0.35


def structural_confidence_score(p):
    return (
        0.35 * p.get("start_boundary_confidence", 0.0)
        + 0.35 * p.get("end_boundary_confidence", 0.0)
        + 0.30 * p.get("phrase_grid_confidence", 0.0)
    )


def role_compatibility_score(out_role, in_role):
    """Score exact structural-role matches above musically sensible neighbors."""
    if out_role == in_role and out_role not in {"unknown", "transition"}:
        return 5.0
    near_matches = {
        "chorus": {"post-chorus": 2.5, "pre-chorus": 1.5, "instrumental": 1.0},
        "post-chorus": {"chorus": 2.5, "instrumental": 1.5, "bridge": 0.8},
        "pre-chorus": {"verse": 2.0, "chorus": 1.5, "bridge": 1.0},
        "verse": {"pre-chorus": 2.0, "bridge": 1.5, "intro": 1.0},
        "bridge": {"verse": 1.5, "pre-chorus": 1.0, "instrumental": 1.0},
        "instrumental": {"chorus": 1.0, "post-chorus": 1.5, "bridge": 1.0},
        "intro": {"verse": 1.5, "instrumental": 1.0},
        "outro": {"post-chorus": 1.0, "instrumental": 1.0},
    }
    return near_matches.get(out_role, {}).get(in_role, -1.5)


def role_sequence_compatibility_score(out_p, in_p):
    """Score exact or element-wise role alignment for single and combined spans."""
    out_roles = phrase_role_sequence(out_p)
    in_roles = phrase_role_sequence(in_p)
    if out_roles == in_roles and out_roles != ("unknown",):
        return 5.0
    if len(out_roles) == len(in_roles):
        return float(np.mean([
            role_compatibility_score(out_role, in_role)
            for out_role, in_role in zip(out_roles, in_roles)
        ]))
    return 0.6 * max(
        role_compatibility_score(out_role, in_role)
        for out_role in out_roles
        for in_role in in_roles
    )


def phrase_energy(p):
    """Read the v2 normalized phrase energy value."""
    value = p.get("energy_percentile")
    if value is None:
        value = p.get("energy_level", 0.5)
    return float(value)


def phrase_vocal_strength(p):
    """Return the strongest normalized vocal-level indicator available."""
    return float(np.clip(max(
        vocal_occ_fraction(p),
        float(p.get("vocal_activity_level", 0.0)),
    ), 0.0, 1.0))


def transition_phrase_dynamics(phrase, phrases):
    """Measure contextual energy dips, lifts, and vocal fade opportunities."""
    ordered = sorted(phrases, key=phrase_start)
    index = next((
        i for i, candidate in enumerate(ordered)
        if candidate is phrase or (
            candidate.get("start_bar") == phrase.get("start_bar")
            and candidate.get("end_bar") == phrase.get("end_bar")
        )
    ), None)
    if index is None:
        return {
            "energy_dip": 0.0,
            "energy_rise": 0.0,
            "vocal_peak": phrase_vocal_strength(phrase),
            "vocal_drop_after": 0.0,
            "vocal_fade_opportunity": 0.5 * phrase_vocal_strength(phrase),
        }

    current_energy = phrase_energy(phrase)
    current_vocal = phrase_vocal_strength(phrase)
    previous = ordered[index - 1] if index > 0 else None
    following = ordered[index + 1] if index + 1 < len(ordered) else None
    previous_energy = phrase_energy(previous) if previous else current_energy
    following_energy = phrase_energy(following) if following else 0.0
    previous_vocal = phrase_vocal_strength(previous) if previous else current_vocal
    following_vocal = phrase_vocal_strength(following) if following else 0.0

    energy_drop_into = max(0.0, previous_energy - current_energy)
    energy_drop_after = max(0.0, current_energy - following_energy)
    internal_energy_drop = max(0.0, -float(phrase.get("energy_trend", 0.0)))
    energy_dip = max(energy_drop_into, energy_drop_after, internal_energy_drop)
    energy_rise = max(
        0.0,
        current_energy - previous_energy,
        float(phrase.get("energy_trend", 0.0)),
    )
    vocal_drop_after = max(0.0, current_vocal - following_vocal)
    vocal_rise_into = max(0.0, current_vocal - previous_vocal)
    vocal_fade_opportunity = current_vocal * (0.55 + 0.45 * vocal_drop_after)

    return {
        "energy_dip": float(np.clip(energy_dip, 0.0, 1.0)),
        "energy_rise": float(np.clip(energy_rise, 0.0, 1.0)),
        "vocal_peak": current_vocal,
        "vocal_drop_after": float(np.clip(vocal_drop_after, 0.0, 1.0)),
        "vocal_rise_into": float(np.clip(vocal_rise_into, 0.0, 1.0)),
        "vocal_fade_opportunity": float(np.clip(vocal_fade_opportunity, 0.0, 1.0)),
    }


def phrase_match_components(out_p, in_p):
    """Return confidence, role, energy, duration, and vocal match components."""
    out_bars = phrase_bars(out_p)
    in_bars = phrase_bars(in_p)
    duration_similarity = 1.0 - abs(out_bars - in_bars) / max(out_bars, in_bars, 1.0)
    return {
        "role_compatibility": role_sequence_compatibility_score(out_p, in_p),
        "phrase_confidence": 3.0 * (
            float(out_p.get("phrase_confidence", 0.0))
            + float(in_p.get("phrase_confidence", 0.0))
        ),
        "structural_confidence": 2.0 * (
            structural_confidence_score(out_p) + structural_confidence_score(in_p)
        ),
        "role_confidence": float(out_p.get("role_confidence", 0.0))
        + float(in_p.get("role_confidence", 0.0)),
        "energy_similarity": 2.0 * (1.0 - abs(phrase_energy(out_p) - phrase_energy(in_p))),
        "duration_similarity": 2.0 * duration_similarity,
        "incoming_vocal_occupancy": 1.5 * vocal_occ_fraction(in_p),
    }


def phrase_match_score(out_p, in_p):
    """Score a phrase pair using v3 structure plus render-time vocal fields."""
    return sum(phrase_match_components(out_p, in_p).values())


def get_renderable_phrases(phrases, allow_edge_roles=False):
    usable = [
        p for p in phrases
        if round(phrase_bars(p)) >= MIN_RENDER_BARS
        and (allow_edge_roles or phrase_role(p) not in FORBIDDEN_ROLES)
        and start_is_acceptable(p)
        and end_is_acceptable(p)
        and p.get("phrase_grid_confidence", 0.0) >= 0.35
    ]
    if usable:
        return usable

    relaxed = [
        p for p in phrases
        if (allow_edge_roles or phrase_role(p) not in FORBIDDEN_ROLES)
        and round(phrase_bars(p)) >= MIN_RENDER_BARS
    ]
    return relaxed


def combine_phrase_span(phrases):
    """Aggregate contiguous phrases into one confidence-aware mashup candidate."""
    bars = np.asarray([phrase_bars(phrase) for phrase in phrases], dtype=float)
    total_bars = float(np.sum(bars))
    weights = bars / max(total_bars, 1e-8)
    confidences = np.asarray([
        float(phrase.get("phrase_confidence", 0.0)) for phrase in phrases
    ])
    roles = [phrase_role(phrase) for phrase in phrases]
    dominant_role = roles[int(np.argmax(bars))]
    remaining_vocal_bars = min(total_bars, MIXER_OVERLAP_TARGET_BARS)
    vocal_parts = []
    for phrase in phrases:
        measured_bars = min(phrase_bars(phrase), remaining_vocal_bars)
        if measured_bars <= 0:
            break
        vocal_parts.append((phrase, measured_bars))
        remaining_vocal_bars -= measured_bars
    measured_total = sum(measured_bars for _, measured_bars in vocal_parts)
    vocal_occupancy = sum(
        measured_bars * phrase_vocal_prefix_metric(phrase, measured_bars, "vocal_occupancy")
        for phrase, measured_bars in vocal_parts
    ) / max(measured_total, 1e-8)
    vocal_activity_level = sum(
        measured_bars * phrase_vocal_prefix_metric(
            phrase, measured_bars, "vocal_activity_level"
        )
        for phrase, measured_bars in vocal_parts
    ) / max(measured_total, 1e-8)
    vocal_rms = np.sqrt(sum(
        measured_bars * phrase_vocal_prefix_metric(phrase, measured_bars, "vocal_rms") ** 2
        for phrase, measured_bars in vocal_parts
    ) / max(measured_total, 1e-8))
    return {
        "start_time": phrase_start(phrases[0]),
        "end_time": phrase_end(phrases[-1]),
        "start_bar": phrases[0].get("start_bar"),
        "end_bar": phrases[-1].get("end_bar"),
        "bar_count": total_bars,
        "phrase_role": dominant_role,
        "phrase_role_sequence": roles,
        "source_phrase_count": len(phrases),
        "source_phrase_ranges": [
            [phrase.get("start_bar"), phrase.get("end_bar")] for phrase in phrases
        ],
        "phrase_confidence": float(0.6 * np.average(confidences, weights=bars) + 0.4 * np.min(confidences)),
        "role_confidence": float(np.average([
            phrase.get("role_confidence", 0.0) for phrase in phrases
        ], weights=bars)),
        "start_boundary_confidence": phrases[0].get("start_boundary_confidence", 0.0),
        "end_boundary_confidence": phrases[-1].get("end_boundary_confidence", 0.0),
        "phrase_grid_confidence": float(np.average([
            phrase.get("phrase_grid_confidence", 0.0) for phrase in phrases
        ], weights=bars)),
        "energy_percentile": float(np.average([
            phrase_energy(phrase) for phrase in phrases
        ], weights=bars)),
        "vocal_occupancy": vocal_occupancy,
        "vocal_activity_level": vocal_activity_level,
        "vocal_rms": float(vocal_rms),
        "vocal_analysis_bars": float(measured_total),
        "vocal_level": "high" if vocal_occupancy >= MIXER_MIN_INCOMING_VOCAL_OCCUPANCY else "low",
        "position": float(np.average([
            phrase.get("position", 0.0) for phrase in phrases
        ], weights=bars)),
        "combined_phrase_span": len(phrases) > 1,
    }


def build_mashup_phrase_spans(phrases, strict=True):
    """Build 8-16 bar candidates, combining only confident contiguous phrases."""
    spans = []
    ordered = sorted(phrases, key=phrase_start)
    for start_index in range(len(ordered)):
        members = []
        total_bars = 0.0
        for index in range(start_index, min(len(ordered), start_index + MIXER_MAX_COMBINED_PHRASES)):
            phrase = ordered[index]
            if phrase_role(phrase) in FORBIDDEN_ROLES or phrase_bars(phrase) <= 0:
                break
            if members and abs(phrase_start(phrase) - phrase_end(members[-1])) > 0.10:
                break
            if members and float(phrase.get("phrase_confidence", 0.0)) < MIXER_COMBINE_MIN_PHRASE_CONFIDENCE:
                break
            if members and float(members[-1].get("phrase_confidence", 0.0)) < MIXER_COMBINE_MIN_PHRASE_CONFIDENCE:
                break
            members.append(phrase)
            total_bars += phrase_bars(phrase)
            if total_bars > MIXER_MAX_OVERLAP_BARS:
                break
            if total_bars < MIN_RENDER_BARS:
                continue

            span = combine_phrase_span(members)
            span["vocal_preroll_occupancy"] = float(np.clip(
                members[0].get("vocal_preroll_occupancy", 0.0),
                0.0,
                1.0,
            ))
            span["vocal_preroll_rms"] = max(
                0.0,
                float(members[0].get("vocal_preroll_rms", 0.0)),
            )
            if strict and (
                not start_is_acceptable(span)
                or not end_is_acceptable(span)
                or span["phrase_grid_confidence"] < 0.35
            ):
                continue
            spans.append(span)
    return spans


def get_track_duration_sec(bundle):
    metadata = bundle.get("metadata") or {}
    dur = metadata.get("duration_sec")
    if dur:
        return float(dur)

    analysis = bundle.get("analysis") or {}
    downbeats = analysis.get("downbeats") or []
    beats = analysis.get("beats") or []
    if downbeats:
        return float(downbeats[-1]) + 8.0
    if beats:
        return float(beats[-1]) + 8.0
    raise ValueError(f"No concrete duration available for '{bundle.get('song_name', 'track')}'")


def select_transition_pair(outgoing_bundle, incoming_bundle, out_bar_dur, tempo_ratio,
                            solo_bars=TRANSITION_SOLO_BARS,
                            crossfade_bars=TRANSITION_CROSSFADE_BARS,
                            return_details=False):
    out_candidates = get_renderable_phrases(
        outgoing_bundle["analysis"]["phrases"], allow_edge_roles=True
    )
    in_candidates = get_renderable_phrases(
        incoming_bundle["analysis"]["phrases"], allow_edge_roles=True
    )

    if not out_candidates or not in_candidates:
        raise ValueError("No renderable phrases available for transition")

    outgoing_duration_sec = get_track_duration_sec(outgoing_bundle)
    incoming_duration_sec = get_track_duration_sec(incoming_bundle)
    planned_bars = crossfade_bars + solo_bars
    planned_sec = planned_bars * out_bar_dur

    valid_out = [
        p for p in out_candidates
        if phrase_end(p) - planned_sec >= 0
        and phrase_end(p) <= outgoing_duration_sec
        and phrase_end(p) - phrase_start(p) >= MIN_RENDER_BARS * out_bar_dur * 0.85
    ]
    valid_in = [
        p for p in in_candidates
        if phrase_start(p) + planned_sec * tempo_ratio <= incoming_duration_sec
        and phrase_end(p) - phrase_start(p) >= MIN_RENDER_BARS * out_bar_dur * tempo_ratio * 0.85
    ]

    if not valid_out:
        raise ValueError("No outgoing phrase has a complete transition window")
    if not valid_in:
        raise ValueError("No incoming phrase has enough following audio for transition")

    best_pair = None
    best_score = -1e9
    best_components = None

    out_role_priority = {
        "post-chorus": 3.0, "chorus": 2.5, "instrumental": 2.0,
        "outro": 1.5, "bridge": 1.0, "pre-chorus": 0.5,
    }
    in_role_priority = {
        "intro": 3.0, "verse": 2.5, "chorus": 2.0,
        "instrumental": 1.5, "pre-chorus": 1.0,
    }

    for out_p in valid_out:
        for in_p in valid_in:
            components = phrase_match_components(out_p, in_p)
            outgoing_dynamics = transition_phrase_dynamics(
                out_p, outgoing_bundle["analysis"]["phrases"]
            )
            incoming_dynamics = transition_phrase_dynamics(
                in_p, incoming_bundle["analysis"]["phrases"]
            )
            components.update({
                "outgoing_role_priority": out_role_priority.get(phrase_role(out_p), 0.0),
                "incoming_role_priority": in_role_priority.get(phrase_role(in_p), 0.0),
                "outgoing_late_position": 2.0 * float(out_p.get("position", 0.0)),
                "outgoing_end_confidence": 1.5 * float(
                    out_p.get("end_boundary_confidence", 0.0)
                ),
                "incoming_start_confidence": 1.5 * float(
                    in_p.get("start_boundary_confidence", 0.0)
                ),
                "outgoing_energy_dip": 3.0 * outgoing_dynamics["energy_dip"],
                "outgoing_vocal_fade_peak": 2.2 * outgoing_dynamics["vocal_fade_opportunity"],
                "outgoing_vocal_drop_after": 1.0 * outgoing_dynamics["vocal_drop_after"],
                "outgoing_low_vocal_space": 1.0 * (1.0 - outgoing_dynamics["vocal_peak"]),
                "incoming_energy_lift": 1.5 * incoming_dynamics["energy_rise"],
                "incoming_vocal_lift": 0.8 * incoming_dynamics["vocal_rise_into"],
            })
            score = sum(components.values())

            if score > best_score:
                best_score = score
                best_pair = (out_p, in_p)
                best_components = components

    if return_details:
        return best_pair[0], best_pair[1], best_score, best_components
    return best_pair


def find_best_overlap_pair(outgoing_phrases, incoming_phrases,
                           target_bars=MIXER_OVERLAP_TARGET_BARS,
                           out_bar_dur=None, tempo_ratio=1.0,
                           return_details=False):
    out_candidates = build_mashup_phrase_spans(outgoing_phrases, strict=True)
    in_candidates = [
        phrase for phrase in build_mashup_phrase_spans(incoming_phrases, strict=True)
        if vocal_occ_fraction(phrase) >= MIXER_MIN_INCOMING_VOCAL_OCCUPANCY
        and phrase.get("vocal_preroll_occupancy", 0.0) >= MIXER_MIN_PREROLL_VOCAL_OCCUPANCY
        and phrase.get("vocal_rms", 0.0) >= MIXER_MIN_INCOMING_VOCAL_RMS
        and phrase.get("vocal_preroll_rms", 0.0) >= MIXER_MIN_PREROLL_VOCAL_RMS
    ]
    if not out_candidates:
        out_candidates = build_mashup_phrase_spans(outgoing_phrases, strict=False)
    if not in_candidates:
        in_candidates = [
            phrase for phrase in build_mashup_phrase_spans(incoming_phrases, strict=False)
            if vocal_occ_fraction(phrase) >= MIXER_MIN_INCOMING_VOCAL_OCCUPANCY
            and phrase.get("vocal_preroll_occupancy", 0.0) >= MIXER_MIN_PREROLL_VOCAL_OCCUPANCY
            and phrase.get("vocal_rms", 0.0) >= MIXER_MIN_INCOMING_VOCAL_RMS
            and phrase.get("vocal_preroll_rms", 0.0) >= MIXER_MIN_PREROLL_VOCAL_RMS
        ]

    if not out_candidates:
        raise ValueError("No renderable phrases available for mixer (outgoing track)")
    if not in_candidates:
        raise ValueError(
            "Incoming track has no phrase with substantial vocal presence "
            f"(vocal_occupancy >= {MIXER_MIN_INCOMING_VOCAL_OCCUPANCY}); "
            "cannot build a mixer overlap "
            "that hands vocal duties to the incoming track"
        )

    if out_bar_dur is not None:
        out_candidates = [
            p for p in out_candidates
            if phrase_start(p) >= LEAD_IN_BARS * out_bar_dur
            and phrase_end(p) - phrase_start(p) >= MIN_RENDER_BARS * out_bar_dur * 0.85
        ]
        in_candidates = [
            p for p in in_candidates
            if phrase_start(p) >= out_bar_dur * tempo_ratio
            and phrase_end(p) - phrase_start(p)
            >= MIN_RENDER_BARS * out_bar_dur * tempo_ratio * 0.85
        ]
        if not out_candidates:
            raise ValueError("No mixer phrase has a complete 4-bar outgoing lead-in")
        if not in_candidates:
            raise ValueError("No incoming mixer phrase has a complete 1-bar vocal pre-roll")

    best_pair = None
    best_score = -1e9
    best_components = None

    for out_p in out_candidates:
        for in_p in in_candidates:
            out_bars = round(phrase_bars(out_p))
            in_bars = round(phrase_bars(in_p))
            shared_bars = min(out_bars, in_bars, target_bars)
            if shared_bars < MIN_RENDER_BARS:
                continue
            components = phrase_match_components(out_p, in_p)
            components["target_overlap"] = -0.75 * (
                abs(out_bars - target_bars) + abs(in_bars - target_bars)
            )
            components["combined_span_confidence"] = 0.5 * (
                float(out_p.get("phrase_confidence", 0.0))
                + float(in_p.get("phrase_confidence", 0.0))
            )
            components["incoming_vocal_preroll"] = float(
                in_p.get("vocal_preroll_occupancy", 0.0)
            )
            score = sum(components.values())

            if score > best_score:
                best_score = score
                best_pair = (out_p, in_p)
                best_components = components

    if best_pair is None:
        raise ValueError("No phrase pair supports an 8-16 bar mixer overlap")
    if return_details:
        return best_pair[0], best_pair[1], best_score, best_components
    return best_pair


# ---------------------------------------------------------------------------
# Song folder discovery
# ---------------------------------------------------------------------------

def list_song_dirs(data_dir):
    return sorted(p for p in Path(data_dir).iterdir() if p.is_dir())


def find_song_dir(song_identifier, data_dir):
    """Resolve a folder name, compatibility song ID, or Spotify track ID."""
    candidate = Path(data_dir) / song_identifier
    if candidate.is_dir():
        return candidate

    dirs = list_song_dirs(data_dir)
    lowered = song_identifier.strip().lower()

    exact_ci = [p for p in dirs if p.name.lower() == lowered]
    if len(exact_ci) == 1:
        return exact_ci[0]

    metadata_matches = []
    for song_dir in dirs:
        metadata_path = song_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = load_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            continue
        identifiers = {
            str(metadata.get("track_id", "")).strip().lower(),
            str(metadata.get("track_uri", "")).strip().lower(),
        }
        if lowered in identifiers:
            metadata_matches.append(song_dir)
    if len(metadata_matches) == 1:
        return metadata_matches[0]
    if len(metadata_matches) > 1:
        matches = ", ".join(path.name for path in metadata_matches)
        raise ValueError(f"Ambiguous track ID '{song_identifier}'. Matches: {matches}")

    partial = [p for p in dirs if lowered in p.name.lower()]
    if len(partial) == 1:
        return partial[0]

    if not partial:
        raise FileNotFoundError(
            f"No song folder or track ID found for '{song_identifier}' in {data_dir}"
        )

    matches = ", ".join(p.name for p in partial[:10])
    raise ValueError(f"Ambiguous song identifier '{song_identifier}'. Matches: {matches}")


def find_main_wav(song_dir):
    wavs = [p for p in song_dir.glob("*.wav") if p.is_file()]
    wavs = [p for p in wavs if p.parent.name != "stems"]
    if len(wavs) == 1:
        return wavs[0]

    preferred = [p for p in wavs if p.stem == song_dir.name]
    if len(preferred) == 1:
        return preferred[0]

    if not wavs:
        raise FileNotFoundError(f"No main WAV found in {song_dir}")

    names = ", ".join(p.name for p in wavs)
    raise ValueError(f"Multiple WAVs found in {song_dir}; cannot infer main track: {names}")


def find_named_json(song_dir, keyword):
    candidates = sorted(
        p for p in song_dir.glob("*.json")
        if keyword.lower() in p.name.lower()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    names = ", ".join(p.name for p in candidates)
    raise ValueError(f"Multiple '{keyword}' JSON files found in {song_dir}: {names}")


def find_stem_paths(song_dir, source_audio=None):
    stems_dir = song_dir / "stems"
    if not stems_dir.is_dir():
        raise FileNotFoundError(f"Stems directory not found: {stems_dir}")

    expected_stems = ("vocals", "drums", "bass", "other")
    stem_paths = {}
    for stem_name in expected_stems:
        stem_file = stems_dir / f"{stem_name}.wav"
        if stem_file.is_file():
            stem_paths[stem_name] = str(stem_file)

    missing = [name for name in expected_stems if name not in stem_paths]
    if missing:
        raise FileNotFoundError(
            f"Stem separation is incomplete in {stems_dir}; missing: "
            + ", ".join(f"{name}.wav" for name in missing)
        )
    if source_audio is not None:
        source_mtime = Path(source_audio).stat().st_mtime_ns
        stale = [
            name for name, path in stem_paths.items()
            if Path(path).stat().st_mtime_ns < source_mtime
        ]
        if stale:
            raise ValueError(
                f"Stem separation is stale for {song_dir.name}; rerun stage 4"
            )

    return stem_paths


def build_song_bundle(song_name, data_dir):
    song_dir = find_song_dir(song_name, data_dir)
    audio_path = find_main_wav(song_dir)
    stem_paths = find_stem_paths(song_dir, audio_path)

    analysis_path = find_named_json(song_dir, "analysis")
    if analysis_path is None:
        raise FileNotFoundError(f"No analysis.json found in {song_dir}")
    if analysis_path.stat().st_mtime_ns < audio_path.stat().st_mtime_ns:
        raise ValueError(f"{analysis_path} is older than its source WAV; rerun analyze.py")
    analysis = load_json(analysis_path)
    if analysis.get("analysis_source") != "original_mix":
        raise ValueError(
            f"{analysis_path} must be regenerated by analyze.py before rendering"
        )
    if int(analysis.get("schema_version", 0)) < 3:
        raise ValueError(f"Unsupported pre-v3 analysis schema in {analysis_path}")
    vocal_analysis = refresh_vocal_occurrence(analysis, stem_paths["vocals"])

    metadata_path = find_named_json(song_dir, "metadata")
    metadata = load_json(metadata_path) if metadata_path else {}

    popularity = float(metadata.get("popularity", 0.0)) if metadata else 0.0

    return {
        "song_name": song_dir.name,
        "data_dir": str(Path(data_dir)),
        "song_dir": str(song_dir),
        "audio_path": str(audio_path),
        "stem_paths": stem_paths,
        "analysis_path": str(analysis_path),
        "analysis": analysis,
        "vocal_analysis": vocal_analysis,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata": metadata,
        "popularity": popularity,
        "tempo": float(analysis.get("tempo", 0.0)),
        "song_key": analysis.get("song_key"),
        "song_scale": analysis.get("song_scale"),
        "energy": float(analysis.get("energy", 0.5)),
    }


# ---------------------------------------------------------------------------
# Song selection: most popular outgoing, best precomputed compatibility incoming
# ---------------------------------------------------------------------------

def load_compatibility_catalog(data_dir):
    """Load and cache the canonical global compatibility catalog by path."""
    path = (Path(data_dir) / COMPATIBILITY_FILENAME).resolve()
    cache_key = str(path)
    if cache_key not in _COMPATIBILITY_CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"Global compatibility catalog not found: {path}")
        catalog = load_json(path)
        if not isinstance(catalog.get("compatible_mixes"), list):
            raise ValueError(f"Invalid global compatibility catalog: {path}")
        _COMPATIBILITY_CACHE[cache_key] = catalog
    return _COMPATIBILITY_CACHE[cache_key]


def compatibility_entries_for_song(song_name, data_dir):
    """Return canonical pairs containing a song, independent of pair orientation."""
    entries = []
    for entry in load_compatibility_catalog(data_dir)["compatible_mixes"]:
        a_name = (entry.get("song_a") or {}).get("song_id")
        b_name = (entry.get("song_b") or {}).get("song_id")
        if song_name == a_name or song_name == b_name:
            entries.append(entry)
    return entries


def compatibility_candidate(entry, outgoing_name):
    """Return the opposite endpoint metadata from an orientation-independent pair."""
    song_a = entry.get("song_a") or {}
    song_b = entry.get("song_b") or {}
    if song_a.get("song_id") == outgoing_name:
        return song_b
    if song_b.get("song_id") == outgoing_name:
        return song_a
    return None


def lookup_compatibility_entry(outgoing_bundle, incoming_song_name):
    """Look up canonical global metadata for an unordered song pair."""
    for entry in compatibility_entries_for_song(
        outgoing_bundle["song_name"], outgoing_bundle["data_dir"]
    ):
        candidate = compatibility_candidate(entry, outgoing_bundle["song_name"])
        if candidate and candidate.get("song_id") == incoming_song_name:
            return entry
    return None


def compatibility_provenance(entry):
    """Extract render-safe score and provenance fields from a canonical pair."""
    if entry is None:
        return {
            "source": COMPATIBILITY_FILENAME,
            "matched": False,
            "rank": None,
            "score_components": None,
            "hard_filter_metrics": None,
        }
    return {
        "source": COMPATIBILITY_FILENAME,
        "matched": True,
        "rank": entry.get("rank"),
        "score_components": entry.get("score_components"),
        "hard_filter_metrics": entry.get("hard_filter_metrics"),
    }

def select_most_popular_song(data_dir, exclude_dirs=None):
    exclude_dirs = exclude_dirs or set()
    best_name = None
    best_popularity = -1.0

    for song_dir in list_song_dirs(data_dir):
        if str(song_dir) in exclude_dirs:
            continue
        metadata_path = find_named_json(song_dir, "metadata")
        if metadata_path is None:
            continue
        try:
            metadata = load_json(metadata_path)
        except Exception:
            continue

        popularity = float(metadata.get("popularity", -1.0))
        if popularity > best_popularity:
            best_popularity = popularity
            best_name = song_dir.name

    if best_name is None:
        raise FileNotFoundError(f"No songs with readable metadata.json found in {data_dir}")

    return best_name


def select_best_compatible_song(outgoing_bundle, data_dir):
    valid_song_names = {p.name for p in list_song_dirs(data_dir)}
    best_entry = None
    best_score = -1.0
    best_song_id = None
    for entry in compatibility_entries_for_song(outgoing_bundle["song_name"], data_dir):
        candidate = compatibility_candidate(entry, outgoing_bundle["song_name"])
        song_id = candidate.get("song_id") if candidate else None
        if song_id not in valid_song_names:
            continue
        score = float(entry.get("compatibility_score", -1.0))
        if score > best_score:
            best_score = score
            best_entry = entry
            best_song_id = song_id

    if best_entry is None:
        raise FileNotFoundError(
            f"No global compatibility pair for '{outgoing_bundle['song_name']}' "
            "matched an existing song folder"
        )

    return best_song_id, best_score

def select_most_popular_compatible_song(outgoing_bundle, data_dir):
    valid_song_names = {p.name for p in list_song_dirs(data_dir)}
    best_entry = None
    best_song_id = None
    best_popularity = -1.0
    for entry in compatibility_entries_for_song(outgoing_bundle["song_name"], data_dir):
        candidate = compatibility_candidate(entry, outgoing_bundle["song_name"])
        song_id = candidate.get("song_id") if candidate else None
        if song_id not in valid_song_names:
            continue
        if song_id == outgoing_bundle["song_name"]:
            continue

        popularity = candidate.get("popularity_score", -1.0)
        popularity = float(popularity) if popularity is not None else -1.0

        if popularity > best_popularity:
            best_popularity = popularity
            best_entry = entry
            best_song_id = song_id

    if best_entry is None:
        raise FileNotFoundError(
            f"No global compatibility pair for '{outgoing_bundle['song_name']}' "
            "matched an existing, distinct song folder"
        )

    return best_song_id, best_popularity


def lookup_compatibility_score(outgoing_bundle, incoming_song_name):
    entry = lookup_compatibility_entry(outgoing_bundle, incoming_song_name)
    score = entry.get("compatibility_score") if entry else None
    return float(score) if score is not None else None


# ---------------------------------------------------------------------------
# Mixer (mashup-style overlap): phrase-aligned, 4-bar buildup,
# one-bar lead-in vocal handoff (outgoing muted / incoming isolated + boosted),
# incoming instrumentation NEVER plays during the overlap.
# ---------------------------------------------------------------------------

def render_dj_mixer(outgoing_bundle, incoming_bundle, compatibility_score=None):
    out_analysis = outgoing_bundle["analysis"]
    in_analysis = incoming_bundle["analysis"]

    out_tempo = outgoing_bundle["tempo"]
    in_tempo = incoming_bundle["tempo"]
    tempo_ratio = incoming_tempo_ratio(out_tempo, in_tempo)

    out_bar_dur = bar_duration_seconds(out_tempo)
    out_phrase, in_phrase, selection_score, selection_components = find_best_overlap_pair(
        out_analysis["phrases"],
        in_analysis["phrases"],
        target_bars=MIXER_OVERLAP_TARGET_BARS,
        out_bar_dur=out_bar_dur,
        tempo_ratio=tempo_ratio,
        return_details=True,
    )

    out_phrase_bars = round(phrase_bars(out_phrase))
    in_phrase_bars = round(phrase_bars(in_phrase))
    out_timed_bars = int((phrase_end(out_phrase) - phrase_start(out_phrase)) / out_bar_dur + 0.15)
    in_timed_bars = int(
        (phrase_end(in_phrase) - phrase_start(in_phrase))
        / (out_bar_dur * tempo_ratio) + 0.15
    )
    overlap_bars = min(
        out_phrase_bars, in_phrase_bars, out_timed_bars, in_timed_bars,
        MIXER_OVERLAP_TARGET_BARS,
    )
    if overlap_bars < 8:
        raise ValueError(
            f"No sufficiently large overlap phrase for mixer "
            f"(min 8 bars, got outgoing={out_phrase_bars}, incoming={in_phrase_bars})"
        )
    overlap_start = phrase_start(out_phrase)
    lead_in_start = overlap_start - (LEAD_IN_BARS * out_bar_dur)
    overlap_end = overlap_start + overlap_bars * out_bar_dur

    if lead_in_start < 0:
        raise ValueError("Not enough outgoing audio before the selected phrase for a 4-bar lead-in")

    y_out_vocals = load_single_stem(outgoing_bundle["stem_paths"], "vocals")
    y_out_instrumental = sum_available_stems(outgoing_bundle["stem_paths"], exclude={"vocals"})

    stretched_in_stems = build_stretched_incoming_stems(incoming_bundle, tempo_ratio, sr=SR)
    if "vocals" not in stretched_in_stems:
        raise FileNotFoundError(
            f"Incoming track '{incoming_bundle['song_name']}' has no isolated vocals stem; "
            f"cannot build an incoming-vocals-only mixer overlap"
        )
    y_in_vocals_stretched = stretched_in_stems["vocals"]
    in_start_stretched = phrase_start(in_phrase) / tempo_ratio
    in_end_stretched = in_start_stretched + (overlap_bars * out_bar_dur)
    bar_samples = int(round(out_bar_dur * SR))
    lead_samples = int(round(LEAD_IN_BARS * out_bar_dur * SR))
    overlap_samples = int(round(overlap_bars * out_bar_dur * SR))
    rendered_samples = lead_samples + overlap_samples

    out_instrumental = slice_planned_window(
        y_out_instrumental, SR, lead_in_start, rendered_samples,
        "outgoing mixer instrumentation",
    )
    if y_out_vocals is not None:
        out_vocals = slice_planned_window(
            y_out_vocals, SR, lead_in_start, rendered_samples, "outgoing mixer vocals"
        )
    else:
        out_vocals = np.zeros(rendered_samples, dtype=INTERNAL_DTYPE)

    incoming_source_start = in_start_stretched - out_bar_dur
    incoming_source_samples = bar_samples + overlap_samples
    incoming_vocal_source = slice_planned_window(
        y_in_vocals_stretched, SR, incoming_source_start, incoming_source_samples,
        "incoming mixer vocal pre-roll and phrase",
    )
    incoming_vocals = np.zeros(rendered_samples, dtype=INTERNAL_DTYPE)
    incoming_offset = lead_samples - bar_samples
    incoming_vocals[incoming_offset:] = incoming_vocal_source

    in_vocals_rms = rms_of(incoming_vocal_source[bar_samples:])
    in_vocals_preroll_rms = rms_of(incoming_vocal_source[:bar_samples])
    if in_vocals_preroll_rms < MIXER_MIN_PREROLL_VOCAL_RMS:
        raise ValueError(
            f"Incoming track '{incoming_bundle['song_name']}' has no audible vocal pre-roll "
            f"one bar before the mashup (rms={in_vocals_preroll_rms:.5f} < "
            f"{MIXER_MIN_PREROLL_VOCAL_RMS})"
        )
    if in_vocals_rms < MIXER_MIN_INCOMING_VOCAL_RMS:
        raise ValueError(
            f"Incoming track '{incoming_bundle['song_name']}' vocals are too quiet/absent "
            f"in the selected overlap window (rms={in_vocals_rms:.5f} < "
            f"{MIXER_MIN_INCOMING_VOCAL_RMS}); refusing to render a mixer overlap that would "
            f"leave the listener with no vocals after the outgoing vocals mute out"
        )

    incoming_vocal_boost_gain = db_to_gain(MIXER_INCOMING_VOCAL_BOOST_DB)
    handoff_start = lead_samples - bar_samples
    handoff_end = lead_samples
    out_vocal_gain, in_vocal_gain = overlapping_handoff_envelopes(
        rendered_samples, handoff_start, handoff_end
    )
    muted_out_vocals = apply_envelope(out_vocals, out_vocal_gain)
    boosted_in_vocals = apply_envelope(
        incoming_vocals, in_vocal_gain * incoming_vocal_boost_gain
    )

    rendered = out_instrumental + muted_out_vocals + boosted_in_vocals
    rendered = soft_limit(rendered)
    rendered = normalize_peak(rendered)

    compatibility_entry = lookup_compatibility_entry(
        outgoing_bundle, incoming_bundle["song_name"]
    )
    if compatibility_entry is not None:
        compatibility_score = float(compatibility_entry["compatibility_score"])

    metadata = {
        "mode": "mixer",
        "outgoing_song_name": outgoing_bundle["song_name"],
        "incoming_song_name": incoming_bundle["song_name"],
        "outgoing_song_dir": outgoing_bundle["song_dir"],
        "incoming_song_dir": incoming_bundle["song_dir"],
        "outgoing_popularity": outgoing_bundle["popularity"],
        "incoming_popularity": incoming_bundle["popularity"],
        "compatibility_score": compatibility_score,
        "compatibility_provenance": compatibility_provenance(compatibility_entry),
        "tempo_stretch_ratio": tempo_ratio,
        "tempo_stretched_track": "incoming",
        "maximum_bpm_stretch_delta": MAX_BPM_STRETCH_DELTA,
        "time_stretch_engine": "pyrubberband" if _HAVE_RUBBERBAND else "librosa_phase_vocoder",
        "sample_rate": SR,
        "output_subtype": OUTPUT_SUBTYPE,
        "outgoing_phrase_role": phrase_role(out_phrase),
        "incoming_phrase_role": phrase_role(in_phrase),
        "outgoing_phrase_bars": out_phrase_bars,
        "incoming_phrase_bars": in_phrase_bars,
        "outgoing_phrase_confidence": out_phrase.get("phrase_confidence"),
        "incoming_phrase_confidence": in_phrase.get("phrase_confidence"),
        "phrase_selection_score": selection_score,
        "phrase_selection_components": selection_components,
        "outgoing_analysis_schema": out_analysis.get("schema_version"),
        "incoming_analysis_schema": in_analysis.get("schema_version"),
        "outgoing_analysis_model": out_analysis.get("model_version"),
        "incoming_analysis_model": in_analysis.get("model_version"),
        "outgoing_analysis_source": out_analysis.get("analysis_source"),
        "incoming_analysis_source": in_analysis.get("analysis_source"),
        "outgoing_vocal_analysis": outgoing_bundle["vocal_analysis"],
        "incoming_vocal_analysis": incoming_bundle["vocal_analysis"],
        "locked_overlap_bars": overlap_bars,
        "lead_in_bars": LEAD_IN_BARS,
        "lead_in_start_sec": lead_in_start,
        "overlap_start_sec": overlap_start,
        "overlap_end_sec": overlap_end,
        "incoming_vocals_start_sec": in_start_stretched,
        "incoming_vocals_end_sec": in_end_stretched,
        "incoming_vocal_preroll_source_start_sec": incoming_source_start,
        "incoming_vocal_occupancy": vocal_occ_fraction(in_phrase),
        "incoming_vocal_rms_confirmed": in_vocals_rms,
        "incoming_vocal_preroll_occupancy": in_phrase.get("vocal_preroll_occupancy"),
        "incoming_vocal_preroll_rms_confirmed": in_vocals_preroll_rms,
        "vocal_handoff_start_render_sec": handoff_start / SR,
        "vocal_handoff_end_render_sec": handoff_end / SR,
        "vocal_handoff_overlap": True,
        "vocal_handoff_curve": "simultaneous equal-power crossfade",
        "outgoing_vocals_available": y_out_vocals is not None,
        "incoming_vocals_available": True,
        "outgoing_vocals_muted_at_mashup_start": True,
        "incoming_vocals_full_at_mashup_start": True,
        "incoming_vocals_isolated_only": True,
        "incoming_instrumentation_excluded_from_overlap": True,
        "incoming_vocal_boost_db": MIXER_INCOMING_VOCAL_BOOST_DB,
        "outgoing_instrumental_continues_through_overlap": True,
        "outgoing_phrase_roles": list(phrase_role_sequence(out_phrase)),
        "incoming_phrase_roles": list(phrase_role_sequence(in_phrase)),
        "outgoing_source_phrase_count": out_phrase.get("source_phrase_count", 1),
        "incoming_source_phrase_count": in_phrase.get("source_phrase_count", 1),
        "outgoing_source_phrase_ranges": out_phrase.get("source_phrase_ranges"),
        "incoming_source_phrase_ranges": in_phrase.get("source_phrase_ranges"),
        "rendered_sample_count": len(rendered),
        "rendered_duration_sec": len(rendered) / SR,
        "boundary_confidence_outgoing_start": out_phrase.get("start_boundary_confidence"),
        "boundary_confidence_outgoing_end": out_phrase.get("end_boundary_confidence"),
        "boundary_confidence_incoming_start": in_phrase.get("start_boundary_confidence"),
        "boundary_confidence_incoming_end": in_phrase.get("end_boundary_confidence"),
        "energy_percentile_outgoing": out_phrase.get("energy_percentile"),
        "energy_percentile_incoming": in_phrase.get("energy_percentile"),
    }

    return rendered, metadata


# ---------------------------------------------------------------------------
# DJ transition: outgoing solo (8 bars) -> EQ-curve crossfade (4 bars,
# phrase-aligned; incoming vocals held back to avoid clashing with the
# outgoing track's fading vocals) -> incoming solo (8 bars).
# ---------------------------------------------------------------------------

def render_dj_transition(outgoing_bundle, incoming_bundle, compatibility_score=None):
    out_analysis = outgoing_bundle["analysis"]
    in_analysis = incoming_bundle["analysis"]

    out_tempo = outgoing_bundle["tempo"]
    in_tempo = incoming_bundle["tempo"]
    tempo_ratio = incoming_tempo_ratio(out_tempo, in_tempo)

    out_bar_dur = bar_duration_seconds(out_tempo)

    solo_bars = TRANSITION_SOLO_BARS
    crossfade_bars = TRANSITION_CROSSFADE_BARS

    out_phrase, in_phrase, selection_score, selection_components = select_transition_pair(
        outgoing_bundle, incoming_bundle, out_bar_dur, tempo_ratio,
        solo_bars=solo_bars, crossfade_bars=crossfade_bars,
        return_details=True,
    )

    out_phrase_bars = round(phrase_bars(out_phrase))
    in_phrase_bars = round(phrase_bars(in_phrase))
    outgoing_transition_dynamics = transition_phrase_dynamics(
        out_phrase, out_analysis["phrases"]
    )
    incoming_transition_dynamics = transition_phrase_dynamics(
        in_phrase, in_analysis["phrases"]
    )

    if out_phrase_bars < MIN_RENDER_BARS or in_phrase_bars < MIN_RENDER_BARS:
        raise ValueError("Transition phrases must be at least 8 bars")

    out_end = phrase_end(out_phrase)
    out_crossfade_start = out_end - (crossfade_bars * out_bar_dur)
    out_solo_start = out_crossfade_start - (solo_bars * out_bar_dur)

    if out_solo_start < 0:
        raise ValueError(
            f"Not enough outgoing audio for the {solo_bars}-bar solo lead before the crossfade"
        )

    bar_samples = int(round(out_bar_dur * SR))
    out_solo_samples = int(round(solo_bars * out_bar_dur * SR))
    crossfade_samples = int(round(crossfade_bars * out_bar_dur * SR))
    in_solo_samples = int(round(solo_bars * out_bar_dur * SR))

    y_out_instrumental = sum_available_stems(
        outgoing_bundle["stem_paths"], exclude={"vocals"}
    )
    y_out_vocals = load_single_stem(outgoing_bundle["stem_paths"], "vocals")
    out_instrumental_window = slice_planned_window(
        y_out_instrumental, SR, out_solo_start, out_solo_samples + crossfade_samples,
        "outgoing transition instrumentation",
    )
    if y_out_vocals is not None:
        out_vocal_window = slice_planned_window(
            y_out_vocals, SR, out_solo_start, out_solo_samples + crossfade_samples,
            "outgoing transition vocals",
        )
    else:
        out_vocal_window = np.zeros_like(out_instrumental_window)
    out_solo_segment = (
        out_instrumental_window[:out_solo_samples]
        + out_vocal_window[:out_solo_samples]
    )
    out_crossfade_instrumental = out_instrumental_window[out_solo_samples:]
    out_crossfade_vocals = out_vocal_window[out_solo_samples:]

    stretched_in_stems = build_stretched_incoming_stems(incoming_bundle, tempo_ratio, sr=SR)
    y_in_instrumental_stretched = sum_stem_dict(stretched_in_stems, exclude={"vocals"})
    y_in_vocals_stretched = stretched_in_stems.get("vocals")

    in_crossfade_start = phrase_start(in_phrase) / tempo_ratio
    in_crossfade_end = in_crossfade_start + (crossfade_bars * out_bar_dur)
    in_solo_end = in_crossfade_end + (solo_bars * out_bar_dur)

    incoming_window_samples = crossfade_samples + in_solo_samples
    in_instrumental_window = slice_planned_window(
        y_in_instrumental_stretched, SR, in_crossfade_start, incoming_window_samples,
        "incoming transition instrumentation",
    )
    if y_in_vocals_stretched is not None:
        in_vocal_window = slice_planned_window(
            y_in_vocals_stretched, SR, in_crossfade_start, incoming_window_samples,
            "incoming transition vocals",
        )
    else:
        in_vocal_window = np.zeros_like(in_instrumental_window)

    in_crossfade_instrumental = in_instrumental_window[:crossfade_samples]
    in_crossfade_vocals = in_vocal_window[:crossfade_samples]
    in_solo_instrumental = in_instrumental_window[crossfade_samples:]
    in_solo_vocals = in_vocal_window[crossfade_samples:]

    crossfade_mix = eq_curve_crossfade(
        out_crossfade_instrumental, in_crossfade_instrumental, SR
    )
    outgoing_vocal_gain = fade_envelope(
        crossfade_samples, 0, bar_samples, fade_in=False
    )
    incoming_vocal_gain = fade_envelope(
        crossfade_samples, crossfade_samples - bar_samples, crossfade_samples,
        fade_in=True,
    )
    crossfade_mix += apply_envelope(out_crossfade_vocals, outgoing_vocal_gain)
    crossfade_mix += apply_envelope(in_crossfade_vocals, incoming_vocal_gain)
    in_solo_segment = in_solo_instrumental + in_solo_vocals

    rendered = np.concatenate([out_solo_segment, crossfade_mix, in_solo_segment])
    rendered = soft_limit(rendered)
    rendered = normalize_peak(rendered)

    compatibility_entry = lookup_compatibility_entry(
        outgoing_bundle, incoming_bundle["song_name"]
    )
    if compatibility_entry is not None:
        compatibility_score = float(compatibility_entry["compatibility_score"])

    metadata = {
        "mode": "transition",
        "outgoing_song_name": outgoing_bundle["song_name"],
        "incoming_song_name": incoming_bundle["song_name"],
        "outgoing_song_dir": outgoing_bundle["song_dir"],
        "incoming_song_dir": incoming_bundle["song_dir"],
        "outgoing_popularity": outgoing_bundle["popularity"],
        "incoming_popularity": incoming_bundle["popularity"],
        "compatibility_score": compatibility_score,
        "compatibility_provenance": compatibility_provenance(compatibility_entry),
        "tempo_stretch_ratio": tempo_ratio,
        "tempo_stretched_track": "incoming",
        "maximum_bpm_stretch_delta": MAX_BPM_STRETCH_DELTA,
        "time_stretch_engine": "pyrubberband" if _HAVE_RUBBERBAND else "librosa_phase_vocoder",
        "sample_rate": SR,
        "output_subtype": OUTPUT_SUBTYPE,
        "outgoing_phrase_role": phrase_role(out_phrase),
        "incoming_phrase_role": phrase_role(in_phrase),
        "outgoing_phrase_bars": out_phrase_bars,
        "incoming_phrase_bars": in_phrase_bars,
        "outgoing_phrase_confidence": out_phrase.get("phrase_confidence"),
        "incoming_phrase_confidence": in_phrase.get("phrase_confidence"),
        "phrase_selection_score": selection_score,
        "phrase_selection_components": selection_components,
        "outgoing_transition_dynamics": outgoing_transition_dynamics,
        "incoming_transition_dynamics": incoming_transition_dynamics,
        "outgoing_analysis_schema": out_analysis.get("schema_version"),
        "incoming_analysis_schema": in_analysis.get("schema_version"),
        "outgoing_analysis_model": out_analysis.get("model_version"),
        "incoming_analysis_model": in_analysis.get("model_version"),
        "outgoing_analysis_source": out_analysis.get("analysis_source"),
        "incoming_analysis_source": in_analysis.get("analysis_source"),
        "outgoing_vocal_analysis": outgoing_bundle["vocal_analysis"],
        "incoming_vocal_analysis": incoming_bundle["vocal_analysis"],
        "outgoing_solo_bars": solo_bars,
        "crossfade_bars": crossfade_bars,
        "incoming_solo_bars": solo_bars,
        "outgoing_solo_start_sec": out_solo_start,
        "outgoing_crossfade_start_sec": out_crossfade_start,
        "outgoing_end_sec": out_end,
        "incoming_crossfade_start_sec": in_crossfade_start,
        "incoming_crossfade_end_sec": in_crossfade_end,
        "incoming_solo_end_sec": in_solo_end,
        "outgoing_vocals_available": y_out_vocals is not None,
        "incoming_vocals_available": y_in_vocals_stretched is not None,
        "outgoing_vocal_fade_start_crossfade_sec": 0.0,
        "outgoing_vocal_fade_end_crossfade_sec": bar_samples / SR,
        "incoming_vocal_fade_start_crossfade_sec": (
            crossfade_samples - bar_samples
        ) / SR,
        "incoming_vocal_fade_end_crossfade_sec": crossfade_samples / SR,
        "vocal_envelopes_overlap": False,
        "incoming_vocals_full_and_continuous_at_solo": y_in_vocals_stretched is not None,
        "incoming_instrumental_full_and_continuous_at_solo": True,
        "crossfade_curve": "eq_curve (band-split equal-power, order="
                            f"{EQ_FILTER_ORDER}: bass swap over first "
                            f"{int(EQ_BASS_SWAP_FRACTION * 100)}% of crossfade, "
                            "mid/high equal-power over full crossfade)",
        "eq_low_cutoff_hz": EQ_LOW_CUTOFF_HZ,
        "eq_high_cutoff_hz": EQ_HIGH_CUTOFF_HZ,
        "eq_filter_order": EQ_FILTER_ORDER,
        "rendered_sample_count": len(rendered),
        "rendered_duration_sec": len(rendered) / SR,
        "boundary_confidence_outgoing_start": out_phrase.get("start_boundary_confidence"),
        "boundary_confidence_outgoing_end": out_phrase.get("end_boundary_confidence"),
        "boundary_confidence_incoming_start": in_phrase.get("start_boundary_confidence"),
        "boundary_confidence_incoming_end": in_phrase.get("end_boundary_confidence"),
        "energy_percentile_outgoing": out_phrase.get("energy_percentile"),
        "energy_percentile_incoming": in_phrase.get("energy_percentile"),
    }

    return rendered, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render_one(mode, outgoing_bundle, incoming_bundle, compatibility_score, output_audio=None, output_metadata=None):
    if mode == "transition":
        rendered, metadata = render_dj_transition(outgoing_bundle, incoming_bundle, compatibility_score)
    else:
        rendered, metadata = render_dj_mixer(outgoing_bundle, incoming_bundle, compatibility_score)

    default_audio_path, default_metadata_path = default_output_paths(
        mode, outgoing_bundle["song_name"], incoming_bundle["song_name"]
    )
    audio_path = output_audio or default_audio_path
    metadata_path = output_metadata or default_metadata_path

    save_audio(audio_path, rendered, sr=SR)
    write_metadata(metadata_path, metadata)
    return audio_path, metadata_path
    

def run_full_batch(data_dir):
    """
    No-parameter mode: for every song in data_dir, treat it as the outgoing
    track and pair it with:

      1. its highest-score pair from the global compatibility catalog
      2. optionally, the most popular compatible endpoint in that catalog

    IMPORTANT:
    - The secondary candidate is NOT the most popular song in the whole library.
      It is the most popular endpoint among that song's global compatible pairs.
    - If a song has no compatible candidates, it is skipped entirely.
    - If the highest-compatibility song and highest-popularity compatible song
      are the same track, that pairing is rendered only once.
    - Both 'transition' and 'mixer' modes are rendered for every unique
      (outgoing, incoming) pair.
    """
    song_dirs = list_song_dirs(data_dir)
    if not song_dirs:
        raise FileNotFoundError(f"No song folders found in {data_dir}")

    total_rendered = 0
    total_skipped = 0

    for song_dir in song_dirs:
        outgoing_name = song_dir.name

        try:
            outgoing_bundle = build_song_bundle(outgoing_name, data_dir)
        except Exception as e:
            print(f"[SKIP outgoing] {outgoing_name}: {e}")
            total_skipped += 1
            continue

        incoming_candidates = {}

        try:
            best_compat_name, best_compat_score = select_best_compatible_song(outgoing_bundle, data_dir)
            if best_compat_name != outgoing_name:
                incoming_candidates[best_compat_name] = best_compat_score
        except Exception as e:
            print(f"[WARN best-compatible] {outgoing_name}: {e}")
            print(f"[SKIP no-compatible] {outgoing_name}: no compatibility-based incoming candidate found")
            total_skipped += 1
            continue

        try:
            most_popular_compat_name, most_popular_compat_popularity = \
                select_most_popular_compatible_song(outgoing_bundle, data_dir)
            if most_popular_compat_name != outgoing_name:
                incoming_candidates.setdefault(most_popular_compat_name, None)
        except Exception as e:
            print(f"[WARN most-popular-compatible] {outgoing_name}: {e}")

        incoming_candidates.pop(outgoing_name, None)

        if not incoming_candidates:
            print(f"[SKIP no-candidates] {outgoing_name}: no valid incoming pairing found")
            total_skipped += 1
            continue

        for incoming_name, known_score in incoming_candidates.items():
            try:
                incoming_bundle = build_song_bundle(incoming_name, data_dir)
            except Exception as e:
                print(f"[SKIP incoming] {outgoing_name} -> {incoming_name}: {e}")
                total_skipped += 1
                continue

            compat_score = known_score
            if compat_score is None:
                compat_score = lookup_compatibility_score(outgoing_bundle, incoming_name)

            for mode in ("transition", "mixer"):
                try:
                    audio_path, metadata_path = render_one(
                        mode, outgoing_bundle, incoming_bundle, compat_score
                    )
                    print(f"[OK {mode}] {outgoing_name} -> {incoming_name}  ({audio_path})")
                    total_rendered += 1
                except (ValueError, FileNotFoundError) as e:
                    print(f"[SKIP {mode}] {outgoing_name} -> {incoming_name}: {e}")
                    total_skipped += 1
                    continue
                except Exception as e:
                    print(f"[SKIP {mode}] {outgoing_name} -> {incoming_name}: unexpected error: {e}")
                    total_skipped += 1
                    continue

    print(f"\nBatch complete: {total_rendered} renders written, {total_skipped} skipped.")
    return total_rendered, total_skipped


def run_mode_batch(data_dir, mode):
    """
    Batch mode for a single render mode ('transition' or 'mixer').

    For every song in data_dir, treat it as the outgoing track and pair it with:
      1. the highest-score pair from the global compatibility catalog
      2. optionally, the highest-popularity compatible endpoint in that catalog

    If both candidates are the same song, the pairing is rendered only once.
    Songs with no compatible candidates are skipped.
    """
    if mode not in {"transition", "mixer"}:
        raise ValueError(f"Unsupported batch mode: {mode}")

    song_dirs = list_song_dirs(data_dir)
    if not song_dirs:
        raise FileNotFoundError(f"No song folders found in {data_dir}")

    total_rendered = 0
    total_skipped = 0

    for song_dir in song_dirs:
        outgoing_name = song_dir.name

        try:
            outgoing_bundle = build_song_bundle(outgoing_name, data_dir)
        except Exception as e:
            print(f"[SKIP outgoing] {outgoing_name}: {e}")
            total_skipped += 1
            continue

        incoming_candidates = {}

        try:
            best_compat_name, best_compat_score = select_best_compatible_song(outgoing_bundle, data_dir)
            if best_compat_name != outgoing_name:
                incoming_candidates[best_compat_name] = best_compat_score
        except Exception as e:
            print(f"[WARN best-compatible] {outgoing_name}: {e}")
            print(f"[SKIP no-compatible] {outgoing_name}: no compatibility-based incoming candidate found")
            total_skipped += 1
            continue

        try:
            most_popular_compat_name, _ = select_most_popular_compatible_song(outgoing_bundle, data_dir)
            if most_popular_compat_name != outgoing_name:
                incoming_candidates.setdefault(most_popular_compat_name, None)
        except Exception as e:
            print(f"[WARN most-popular-compatible] {outgoing_name}: {e}")

        incoming_candidates.pop(outgoing_name, None)

        if not incoming_candidates:
            print(f"[SKIP no-candidates] {outgoing_name}: no valid incoming pairing found")
            total_skipped += 1
            continue

        for incoming_name, known_score in incoming_candidates.items():
            try:
                incoming_bundle = build_song_bundle(incoming_name, data_dir)
            except Exception as e:
                print(f"[SKIP incoming] {outgoing_name} -> {incoming_name}: {e}")
                total_skipped += 1
                continue

            compat_score = known_score
            if compat_score is None:
                compat_score = lookup_compatibility_score(outgoing_bundle, incoming_name)

            try:
                audio_path, metadata_path = render_one(
                    mode, outgoing_bundle, incoming_bundle, compat_score
                )
                print(f"[OK {mode}] {outgoing_name} -> {incoming_name}  ({audio_path})")
                total_rendered += 1
            except (ValueError, FileNotFoundError) as e:
                print(f"[SKIP {mode}] {outgoing_name} -> {incoming_name}: {e}")
                total_skipped += 1
                continue
            except Exception as e:
                print(f"[SKIP {mode}] {outgoing_name} -> {incoming_name}: unexpected error: {e}")
                total_skipped += 1
                continue

    print(f"\nBatch complete for mode={mode}: {total_rendered} renders written, {total_skipped} skipped.")
    return total_rendered, total_skipped


def default_output_paths(mode, outgoing_song, incoming_song):
    """Build output paths under an '<outgoing> to <incoming>' pair folder."""
    safe_out = outgoing_song.replace("/", "-")
    safe_in = incoming_song.replace("/", "-")

    if mode == "transition":
        audio_name = f"transition_to_{safe_in}.wav"
        meta_name = f"transition_to_{safe_in}_metadata.json"
    else:
        audio_name = f"mixer_with_{safe_in}.wav"
        meta_name = f"mixer_with_{safe_in}_metadata.json"

    output_dir = DEFAULT_RENDERS_DIR / f"{safe_out} to {safe_in}"
    return output_dir / audio_name, output_dir / meta_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["transition", "mixer"], default=None)
    parser.add_argument(
        "--outgoing-song",
        default=None,
        help="Outgoing folder name, compatibility song ID, or Spotify track ID",
    )
    parser.add_argument(
        "--incoming-song",
        "--incoming-song-id",
        dest="incoming_song",
        default=None,
        help="Incoming folder name, compatibility song ID, or Spotify track ID",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-audio", default=None)
    parser.add_argument("--output-metadata", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not _HAVE_RUBBERBAND:
        warnings.warn(
            "pyrubberband/Rubber Band CLI not found -- falling back to librosa's phase "
            "vocoder for time-stretching, which will sound noticeably lower quality on "
            "vocals and transients. Install with `pip install pyrubberband` plus the "
            "Rubber Band CLI binary for best results."
        )

    # No mode and no explicit songs -> run the full batch across all tracks in both modes.
    if args.mode is None and args.outgoing_song is None and args.incoming_song is None:
        rendered, _ = run_full_batch(data_dir)
        if rendered == 0:
            raise RuntimeError("Render stage produced no outputs")
        return

    # Mode provided, but no explicit songs -> run batch across the full library for that mode only.
    if args.mode is not None and args.outgoing_song is None and args.incoming_song is None:
        rendered, _ = run_mode_batch(data_dir, args.mode)
        if rendered == 0:
            raise RuntimeError(f"Render stage produced no {args.mode} outputs")
        return

    if args.incoming_song:
        incoming_bundle = build_song_bundle(args.incoming_song, data_dir)
        if args.outgoing_song:
            outgoing_bundle = build_song_bundle(args.outgoing_song, data_dir)
            compatibility_score = lookup_compatibility_score(
                outgoing_bundle, incoming_bundle["song_name"]
            )
        else:
            auto_outgoing_name, compatibility_score = select_best_compatible_song(
                incoming_bundle, data_dir
            )
            outgoing_bundle = build_song_bundle(auto_outgoing_name, data_dir)
    else:
        outgoing_bundle = build_song_bundle(args.outgoing_song, data_dir)
        auto_incoming_name, compatibility_score = select_best_compatible_song(outgoing_bundle, data_dir)
        incoming_bundle = build_song_bundle(auto_incoming_name, data_dir)

    output_audio = Path(args.output_audio) if args.output_audio else None
    output_metadata = Path(args.output_metadata) if args.output_metadata else None
    modes = (args.mode,) if args.mode else ("transition", "mixer")
    if len(modes) > 1 and (output_audio is not None or output_metadata is not None):
        raise ValueError(
            "--output-audio and --output-metadata require --mode for a single output"
        )

    print(f"Outgoing: {outgoing_bundle['song_name']}")
    print(f"Incoming: {incoming_bundle['song_name']}")
    if compatibility_score is not None:
        print(f"Compatibility score: {compatibility_score}")

    failures = []
    for mode in modes:
        try:
            audio_path, metadata_path = render_one(
                mode,
                outgoing_bundle,
                incoming_bundle,
                compatibility_score,
                output_audio=output_audio,
                output_metadata=output_metadata,
            )
        except (ValueError, FileNotFoundError) as e:
            print(
                f"[SKIP {mode}] "
                f"{outgoing_bundle['song_name']} -> {incoming_bundle['song_name']}: {e}"
            )
            failures.append((mode, e))
            continue
        print(f"[{mode}] Saved audio: {audio_path}")
        print(f"[{mode}] Saved metadata: {metadata_path}")
    if failures:
        failed_modes = ", ".join(mode for mode, _ in failures)
        raise RuntimeError(f"Render failed for requested mode(s): {failed_modes}")


if __name__ == "__main__":
    main()
