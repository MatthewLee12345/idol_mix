from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import lfilter


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RENDERS_ROOT = Path(os.environ.get("KPOP_DJ_RENDERS_DIR", PROJECT_ROOT / "renders"))
TRANSITIONS_ROOT = RENDERS_ROOT
SONGS_ROOT = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_ROOT = RENDERS_ROOT

VIDEO_W = 1080
VIDEO_H = 1920
FPS = 24
WAVEFORM_POINTS = 192
WAVEFORM_WIDTH = WAVEFORM_POINTS
BASS_SMOOTH = 0.82
PULSE_STRENGTH = 0.08
SHAKE_STRENGTH = 8.0

LATIN_SEMIBOLD_FONT = Path("/Library/Fonts/Quicksand-SemiBold.ttf")
LATIN_BOLD_FONT = Path("/Library/Fonts/Quicksand-Bold.ttf")
UNICODE_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")

TRANSITION_CAPTION = "WAIT FOR THE SWITCH"
MASHUP_CAPTION = "TWO SONGS. ONE MASHUP."
ASSET_MODES = ("transition", "mixer")


@dataclass(frozen=True)
class RenderAsset:
    """Describe one exact WAV and metadata pair for a render mode."""

    mode: str
    wav_path: Path
    metadata_path: Path
    metadata: dict


@dataclass(frozen=True)
class TextLayer:
    """Store a pre-rendered RGB text layer and its alpha channel."""

    rgb: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class CardAssets:
    """Store precomputed cover cards, masks, shadows, and glow layers."""

    cover_a: np.ndarray
    cover_b: np.ndarray
    mask: np.ndarray
    border_rgb: np.ndarray
    border_alpha: np.ndarray
    shadow_rgb: np.ndarray
    shadow_alpha: np.ndarray
    glow_rgb: np.ndarray
    glow_alpha: np.ndarray
    effect_pad: int


@dataclass(frozen=True)
class VideoLayout:
    """Hold deterministic positions for transition and mashup elements."""

    transition_card_x: int
    transition_card_y: int
    transition_card_size: int
    transition_caption_x: int
    transition_caption_y: int
    transition_title_x: int
    transition_title_y: int
    mashup_card_a_x: int
    mashup_card_b_x: int
    mashup_card_y: int
    mashup_card_size: int
    mashup_caption_x: int
    mashup_caption_y: int
    mashup_title_a_x: int
    mashup_title_b_x: int
    mashup_title_y: int
    mashup_x_x: int
    mashup_x_y: int


@dataclass(frozen=True)
class PairContext:
    """Cache all pair-level visual assets shared by both output modes."""

    pair_dir: Path
    width: int
    height: int
    title_a: str
    title_b: str
    song_dir_a: Path
    song_dir_b: Path
    cover_path_a: Path
    cover_path_b: Path
    background_a: np.ndarray
    background_b: np.ndarray
    transition_cards: CardAssets
    mashup_cards: CardAssets
    transition_caption: TextLayer
    transition_title_a: TextLayer
    transition_title_b: TextLayer
    mashup_caption: TextLayer
    mashup_title_a: TextLayer
    mashup_title_b: TextLayer
    mashup_x: TextLayer
    mashup_bridge_alpha: np.ndarray
    layout: VideoLayout


def read_json_object(path: Path) -> dict:
    """Read a JSON object and reject non-object top-level values."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def finite_score(value: object, default: float = -math.inf) -> float:
    """Convert a metadata score to a finite float for stable sorting."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if math.isfinite(score) else default


def asset_order_key(asset: RenderAsset) -> Tuple[float, float, str, str]:
    """Sort higher metadata scores first and filenames ascending on ties."""

    compatibility = finite_score(asset.metadata.get("compatibility_score"))
    phrase = finite_score(asset.metadata.get("phrase_selection_score"))
    name = asset.metadata_path.name
    return -compatibility, -phrase, name.casefold(), name


def paired_wav_path(metadata_path: Path) -> Path:
    """Return the exact WAV path represented by a metadata filename."""

    stem = metadata_path.stem
    if stem.endswith("_metadata"):
        stem = stem[: -len("_metadata")]
    return metadata_path.with_name(f"{stem}.wav")


def find_metadata_wav_pairs(pair_dir: Path) -> List[Tuple[Path, Path, dict]]:
    """Find exact WAV/JSON pairs with an explicit supported metadata mode."""

    if not pair_dir.is_dir():
        return []

    pairs: List[Tuple[Path, Path, dict]] = []
    metadata_paths = sorted(
        (
            path
            for path in pair_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".json"
            and path.name != "render_manifest.json"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    for metadata_path in metadata_paths:
        try:
            metadata = read_json_object(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        mode_value = metadata.get("mode")
        mode = mode_value.strip().lower() if isinstance(mode_value, str) else ""
        if mode not in ASSET_MODES:
            continue
        wav_path = paired_wav_path(metadata_path)
        if wav_path.is_file():
            pairs.append((wav_path, metadata_path, metadata))
    return pairs


def discover_mode_assets(pair_dir: Path) -> Dict[str, Optional[RenderAsset]]:
    """Select exactly one best exact asset pair for each explicit mode."""

    candidates: Dict[str, List[RenderAsset]] = {mode: [] for mode in ASSET_MODES}
    for wav_path, metadata_path, metadata in find_metadata_wav_pairs(pair_dir):
        mode = str(metadata["mode"]).strip().lower()
        candidates[mode].append(
            RenderAsset(
                mode=mode,
                wav_path=wav_path,
                metadata_path=metadata_path,
                metadata=metadata,
            )
        )

    selected: Dict[str, Optional[RenderAsset]] = {}
    for mode in ASSET_MODES:
        ordered = sorted(candidates[mode], key=asset_order_key)
        selected[mode] = ordered[0] if ordered else None
    return selected


def is_transition_metadata(metadata_path: Path, metadata: dict) -> bool:
    """Return true only for metadata explicitly marked as transition mode."""

    del metadata_path
    mode = metadata.get("mode")
    return isinstance(mode, str) and mode.strip().lower() == "transition"


def choose_metadata_for_transition_dir(
    pair_dir: Path,
) -> Tuple[Optional[Path], Optional[dict], Optional[Path]]:
    """Return the selected explicit transition asset in the legacy tuple form."""

    asset = discover_mode_assets(pair_dir)["transition"]
    if asset is None:
        return None, None, None
    return asset.wav_path, asset.metadata, asset.metadata_path


def normalize_text(text: str) -> str:
    """Normalize a song label for deterministic folder matching."""

    normalized = text.strip().casefold().replace("&", " and ")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def discover_song_folders(songs_root: Path) -> List[Path]:
    """List immediate song folders in deterministic order."""

    if not songs_root.is_dir():
        return []
    return sorted(
        (path for path in songs_root.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )


def resolve_song_folder(track_name: str, song_dirs: Sequence[Path]) -> Optional[Path]:
    """Resolve a metadata song label to the best deterministic song folder."""

    query = normalize_text(track_name)
    if not query:
        return None

    exact = [path for path in song_dirs if normalize_text(path.name) == query]
    if exact:
        return sorted(exact, key=lambda path: (path.name.casefold(), path.name))[0]

    scored: List[Tuple[float, str, str, Path]] = []
    for path in song_dirs:
        folder_name = normalize_text(path.name)
        folder_parts = [normalize_text(part) for part in path.name.split(" - ")]
        candidates = [folder_name, *folder_parts]
        score = max(SequenceMatcher(None, query, candidate).ratio() for candidate in candidates)
        scored.append((-score, path.name.casefold(), path.name, path))
    scored.sort()
    if scored and -scored[0][0] >= 0.62:
        return scored[0][3]
    return None


def valid_metadata_song_dir(
    value: object,
    song_dirs: Sequence[Path],
) -> Optional[Path]:
    """Resolve a metadata song directory only when it belongs to the library."""

    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_dir() or not (resolved / "metadata.json").is_file():
        return None
    for song_dir in song_dirs:
        try:
            if resolved == song_dir.resolve():
                return resolved
        except OSError:
            continue
    return None


def resolve_pair_song_folder(
    role: str,
    assets: Sequence[RenderAsset],
    song_dirs: Sequence[Path],
) -> Optional[Path]:
    """Resolve one pair song once, preferring valid metadata directory paths."""

    directory_key = f"{role}_song_dir"
    name_key = f"{role}_song_name"
    for asset in assets:
        direct = valid_metadata_song_dir(asset.metadata.get(directory_key), song_dirs)
        if direct is not None:
            name = asset.metadata.get(name_key)
            if not isinstance(name, str) or not name.strip():
                return direct
            if normalize_text(direct.name) == normalize_text(name):
                return direct
    for asset in assets:
        name = asset.metadata.get(name_key)
        if isinstance(name, str) and name.strip():
            resolved = resolve_song_folder(name, song_dirs)
            if resolved is not None:
                return resolved
    return None


def load_song_metadata(song_dir: Path) -> dict:
    """Load a song metadata file and require its authoritative track name."""

    metadata_path = song_dir / "metadata.json"
    metadata = read_json_object(metadata_path)
    track_name = metadata.get("track_name")
    if not isinstance(track_name, str) or not track_name.strip():
        raise ValueError(f"Song metadata has no authoritative track_name: {metadata_path}")
    return metadata


def find_album_image(song_dir: Path) -> Optional[Path]:
    """Choose a cover image by preferred names and deterministic fallback."""

    preferred_names = (
        "cover.jpg",
        "cover.jpeg",
        "cover.png",
        "album_image.jpg",
        "album_image.jpeg",
        "album_image.png",
    )
    files_by_name = {
        path.name.casefold(): path
        for path in song_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for name in preferred_names:
        if name in files_by_name:
            return files_by_name[name]

    images = sorted(
        files_by_name.values(), key=lambda path: (path.name.casefold(), path.name)
    )
    cover_images = [path for path in images if "cover" in path.stem.casefold()]
    return (cover_images or images or [None])[0]


def load_cover(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load cover image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_fill(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize and center-crop an image to fill target dimensions."""

    source_h, source_w = image.shape[:2]
    if source_h <= 0 or source_w <= 0 or width <= 0 or height <= 0:
        raise ValueError("Image and target dimensions must be positive")
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, int(math.ceil(source_w * scale)))
    resized_h = max(height, int(math.ceil(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)
    left = (resized_w - width) // 2
    top = (resized_h - height) // 2
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


def radial_vignette(width: int, height: int, strength: float = 0.34) -> np.ndarray:
    """Create a smooth edge-darkening mask for a portrait background."""

    yy, xx = np.mgrid[0:height, 0:width]
    dx = (xx - width / 2.0) / max(width / 2.0, 1.0)
    dy = (yy - height * 0.46) / max(height / 2.0, 1.0)
    radius = np.sqrt(dx * dx + dy * dy)
    return np.clip(1.0 - strength * np.clip(radius, 0.0, 1.0) ** 1.35, 0.55, 1.0).astype(
        np.float32
    )


def prepare_background(
    cover: np.ndarray,
    width: int,
    height: int,
    vignette: np.ndarray,
) -> np.ndarray:
    """Pre-render a dark blurred cover background for repeated frame blending."""

    scale = min(width / VIDEO_W, height / VIDEO_H)
    sigma = max(4.0, 34.0 * scale)
    background = cv2.GaussianBlur(crop_fill(cover, width, height), (0, 0), sigma)
    values = background.astype(np.float32)
    values *= (0.57 * vignette)[..., None]
    values += np.array([8.0, 7.0, 16.0], dtype=np.float32)
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Create an antialiased rounded-rectangle alpha mask."""

    radius = max(1, min(radius, width // 2, height // 2))
    supersample = 4
    image = Image.new("L", (width * supersample, height * supersample), 0)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, width * supersample - 1, height * supersample - 1),
        radius=radius * supersample,
        fill=255,
    )
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def build_card_assets(
    cover_a: np.ndarray,
    cover_b: np.ndarray,
    size: int,
    radius: int,
) -> CardAssets:
    """Precompute two same-size cover cards and all reusable effect masks."""

    card_a = crop_fill(cover_a, size, size)
    card_b = crop_fill(cover_b, size, size)
    mask = rounded_mask(size, size, radius)
    border_width = max(2, int(round(size * 0.008)))
    kernel = np.ones((border_width * 2 + 1, border_width * 2 + 1), dtype=np.uint8)
    border_alpha = cv2.subtract(mask, cv2.erode(mask, kernel))
    border_rgb = np.empty_like(card_a)
    border_rgb[:] = (232, 248, 255)

    effect_pad = max(12, int(round(size * 0.10)))
    effect_shape = (size + 2 * effect_pad, size + 2 * effect_pad)
    effect_mask = np.zeros(effect_shape, dtype=np.uint8)
    effect_mask[effect_pad : effect_pad + size, effect_pad : effect_pad + size] = mask
    shadow_alpha = cv2.GaussianBlur(effect_mask, (0, 0), max(4.0, size * 0.035))
    shadow_alpha = np.clip(shadow_alpha.astype(np.float32) * 0.48, 0, 255).astype(np.uint8)
    glow_alpha = cv2.GaussianBlur(effect_mask, (0, 0), max(4.0, size * 0.045))

    shadow_rgb = np.zeros((*effect_shape, 3), dtype=np.uint8)
    glow_rgb = np.empty((*effect_shape, 3), dtype=np.uint8)
    glow_rgb[:] = (116, 220, 255)
    return CardAssets(
        cover_a=card_a,
        cover_b=card_b,
        mask=mask,
        border_rgb=border_rgb,
        border_alpha=border_alpha,
        shadow_rgb=shadow_rgb,
        shadow_alpha=shadow_alpha,
        glow_rgb=glow_rgb,
        glow_alpha=glow_alpha,
        effect_pad=effect_pad,
    )


def compute_layout(width: int, height: int) -> VideoLayout:
    """Compute a balanced portrait layout with short-form safe margins."""

    side_margin = max(12, int(round(width * 0.067)))
    transition_size = min(width - 2 * side_margin, int(round(height * 0.47)))
    transition_x = (width - transition_size) // 2
    transition_y = int(round(height * 0.25))

    mashup_margin = max(10, int(round(width * 0.06)))
    mashup_gap = max(24, int(round(width * 0.10)))
    mashup_size = (width - 2 * mashup_margin - mashup_gap) // 2
    mashup_y = int(round(height * 0.31))
    mashup_a_x = mashup_margin
    mashup_b_x = width - mashup_margin - mashup_size
    title_gap = max(8, int(round(height * 0.02)))

    transition_text_width = width - 2 * side_margin
    mashup_title_width = mashup_size + max(8, int(round(width * 0.02)))
    x_layer_width = max(24, mashup_gap)
    x_layer_height = max(36, int(round(height * 0.12)))
    return VideoLayout(
        transition_card_x=transition_x,
        transition_card_y=transition_y,
        transition_card_size=transition_size,
        transition_caption_x=(width - transition_text_width) // 2,
        transition_caption_y=int(round(height * 0.105)),
        transition_title_x=(width - transition_text_width) // 2,
        transition_title_y=transition_y + transition_size + title_gap,
        mashup_card_a_x=mashup_a_x,
        mashup_card_b_x=mashup_b_x,
        mashup_card_y=mashup_y,
        mashup_card_size=mashup_size,
        mashup_caption_x=side_margin,
        mashup_caption_y=int(round(height * 0.12)),
        mashup_title_a_x=mashup_a_x - (mashup_title_width - mashup_size) // 2,
        mashup_title_b_x=mashup_b_x - (mashup_title_width - mashup_size) // 2,
        mashup_title_y=mashup_y + mashup_size + title_gap,
        mashup_x_x=(width - x_layer_width) // 2,
        mashup_x_y=mashup_y + mashup_size // 2 - x_layer_height // 2,
    )


def font_path_for_text(text: str, bold: bool) -> Path:
    """Choose Quicksand for Latin text and a Unicode font otherwise."""

    if any(ord(character) > 127 for character in text):
        candidates = (UNICODE_FONT, LATIN_BOLD_FONT, LATIN_SEMIBOLD_FONT)
    elif bold:
        candidates = (LATIN_BOLD_FONT, LATIN_SEMIBOLD_FONT, UNICODE_FONT)
    else:
        candidates = (LATIN_SEMIBOLD_FONT, LATIN_BOLD_FONT, UNICODE_FONT)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No configured Pillow title font is available")


def load_text_font(text: str, size: int, bold: bool) -> ImageFont.FreeTypeFont:
    """Load the configured Pillow font suitable for the supplied text."""

    return ImageFont.truetype(str(font_path_for_text(text, bold)), size=max(1, size))


def measure_text(text: str, font: ImageFont.FreeTypeFont) -> float:
    """Measure one text line using Pillow's font metrics."""

    return float(font.getlength(text))


def break_long_token(
    token: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """Split an over-wide token at character boundaries."""

    pieces: List[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and measure_text(candidate, font) > max_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [""]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Greedily wrap text while supporting long unspaced Unicode titles."""

    words = text.strip().split()
    if not words:
        return [""]

    lines: List[str] = []
    current = ""
    for word in words:
        pieces = break_long_token(word, font, max_width)
        for piece_index, piece in enumerate(pieces):
            separator = " " if current and piece_index == 0 else ""
            candidate = f"{current}{separator}{piece}"
            if current and measure_text(candidate, font) > max_width:
                lines.append(current)
                current = piece
            else:
                current = candidate
            if piece_index < len(pieces) - 1:
                lines.append(current)
                current = ""
    if current:
        lines.append(current)
    return lines


def ellipsize_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Trim a line to fit and append an ASCII ellipsis."""

    suffix = "..."
    value = text.rstrip()
    while value and measure_text(value + suffix, font) > max_width:
        value = value[:-1].rstrip()
    return value + suffix if value else suffix


def fit_wrapped_text(
    text: str,
    max_width: int,
    max_lines: int,
    preferred_size: int,
    minimum_size: int,
    bold: bool,
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    """Fit wrapped text by shrinking, then truncate with an ellipsis if needed."""

    final_font = load_text_font(text, minimum_size, bold)
    final_lines = wrap_text(text, final_font, max_width)
    for size in range(preferred_size, minimum_size - 1, -2):
        font = load_text_font(text, size, bold)
        lines = wrap_text(text, font, max_width)
        final_font, final_lines = font, lines
        if len(lines) <= max_lines:
            return font, lines
    if len(final_lines) > max_lines:
        final_lines = final_lines[:max_lines]
        final_lines[-1] = ellipsize_text(final_lines[-1], final_font, max_width)
    return final_font, final_lines


def render_text_layer(
    text: str,
    width: int,
    height: int,
    preferred_size: int,
    minimum_size: int,
    color: Tuple[int, int, int],
    max_lines: int = 2,
    bold: bool = False,
    pill: bool = False,
) -> TextLayer:
    """Pre-render fitted Pillow text with a shadow, stroke, and optional pill."""

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale = max(0.2, min(width / VIDEO_W, height / (VIDEO_H * 0.12)))
    horizontal_padding = max(8, int(round(28 * scale)))
    max_text_width = max(1, width - 2 * horizontal_padding)
    font, lines = fit_wrapped_text(
        text=text,
        max_width=max_text_width,
        max_lines=max_lines,
        preferred_size=preferred_size,
        minimum_size=minimum_size,
        bold=bold,
    )
    stroke_width = max(1, int(round(preferred_size * 0.045)))
    shadow_offset = max(1, int(round(preferred_size * 0.07)))
    sample_box = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_width)
    line_height = max(1, sample_box[3] - sample_box[1])
    spacing = max(2, int(round(font.size * 0.16)))
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
    first_y = (height - total_height) / 2.0 + line_height / 2.0
    line_widths = [measure_text(line, font) for line in lines]

    if pill:
        pill_width = min(width - 2, int(math.ceil(max(line_widths, default=0) + 2 * horizontal_padding)))
        pill_height = min(height - 2, total_height + max(10, int(round(24 * scale))))
        left = (width - pill_width) // 2
        top = (height - pill_height) // 2
        radius = max(6, pill_height // 2)
        draw.rounded_rectangle(
            (left + shadow_offset, top + shadow_offset, left + pill_width, top + pill_height),
            radius=radius,
            fill=(0, 0, 0, 92),
        )
        draw.rounded_rectangle(
            (left, top, left + pill_width, top + pill_height),
            radius=radius,
            fill=(13, 15, 24, 196),
            outline=(255, 255, 255, 38),
            width=max(1, stroke_width // 2),
        )

    for index, line in enumerate(lines):
        center_y = first_y + index * (line_height + spacing)
        position = (width / 2.0, center_y)
        draw.text(
            (position[0] + shadow_offset, position[1] + shadow_offset),
            line,
            font=font,
            anchor="mm",
            fill=(0, 0, 0, 190),
            stroke_width=stroke_width + 1,
            stroke_fill=(0, 0, 0, 190),
        )
        draw.text(
            position,
            line,
            font=font,
            anchor="mm",
            fill=(*color, 255),
            stroke_width=stroke_width,
            stroke_fill=(8, 9, 15, 235),
        )

    rgba = np.asarray(image, dtype=np.uint8)
    return TextLayer(rgb=np.ascontiguousarray(rgba[..., :3]), alpha=np.ascontiguousarray(rgba[..., 3]))


def build_pair_context(
    pair_dir: Path,
    assets: Mapping[str, Optional[RenderAsset]],
    song_dirs: Sequence[Path],
    width: int,
    height: int,
) -> PairContext:
    """Resolve songs once and precompute every shared visual pair asset."""

    selected_assets = [
        asset for mode in ASSET_MODES if (asset := assets.get(mode)) is not None
    ]
    if not selected_assets:
        raise ValueError(f"No supported render assets in {pair_dir}")

    resolved_pairs: List[Tuple[Path, Path]] = []
    for asset in selected_assets:
        song_dir_a = resolve_pair_song_folder("outgoing", [asset], song_dirs)
        song_dir_b = resolve_pair_song_folder("incoming", [asset], song_dirs)
        if song_dir_a is None or song_dir_b is None:
            raise FileNotFoundError(
                f"Could not resolve both song folders from {asset.metadata_path.name}"
            )
        resolved_pairs.append((song_dir_a.resolve(), song_dir_b.resolve()))

    song_dir_a, song_dir_b = resolved_pairs[0]
    if any(pair != resolved_pairs[0] for pair in resolved_pairs[1:]):
        raise ValueError("Selected mode metadata identifies different song pairs")

    song_metadata_a = load_song_metadata(song_dir_a)
    song_metadata_b = load_song_metadata(song_dir_b)
    title_a = str(song_metadata_a["track_name"]).strip()
    title_b = str(song_metadata_b["track_name"]).strip()
    cover_path_a = find_album_image(song_dir_a)
    cover_path_b = find_album_image(song_dir_b)
    if cover_path_a is None or cover_path_b is None:
        raise FileNotFoundError("Could not resolve both deterministic cover images")

    cover_a = load_cover(cover_path_a)
    cover_b = load_cover(cover_path_b)
    layout = compute_layout(width, height)
    vignette = radial_vignette(width, height)
    background_a = prepare_background(cover_a, width, height, vignette)
    background_b = prepare_background(cover_b, width, height, vignette)
    scale = min(width / VIDEO_W, height / VIDEO_H)

    transition_cards = build_card_assets(
        cover_a,
        cover_b,
        layout.transition_card_size,
        max(8, int(round(44 * scale))),
    )
    mashup_cards = build_card_assets(
        cover_a,
        cover_b,
        layout.mashup_card_size,
        max(7, int(round(34 * scale))),
    )

    transition_text_width = width - 2 * layout.transition_caption_x
    transition_caption_height = max(46, int(round(height * 0.10)))
    transition_title_height = max(54, int(round(height * 0.12)))
    mashup_caption_width = width - 2 * layout.mashup_caption_x
    mashup_caption_height = max(44, int(round(height * 0.10)))
    mashup_title_width = layout.mashup_card_size + max(8, int(round(width * 0.02)))
    mashup_title_height = max(54, int(round(height * 0.12)))
    x_layer_width = max(24, layout.mashup_card_b_x - (layout.mashup_card_a_x + layout.mashup_card_size))
    x_layer_height = max(36, int(round(height * 0.12)))

    transition_caption = render_text_layer(
        TRANSITION_CAPTION,
        transition_text_width,
        transition_caption_height,
        max(14, int(round(58 * scale))),
        max(10, int(round(38 * scale))),
        (255, 221, 107),
        max_lines=1,
        bold=True,
        pill=True,
    )
    transition_title_a = render_text_layer(
        title_a,
        transition_text_width,
        transition_title_height,
        max(14, int(round(68 * scale))),
        max(10, int(round(40 * scale))),
        (250, 250, 252),
        max_lines=2,
        bold=False,
    )
    transition_title_b = render_text_layer(
        title_b,
        transition_text_width,
        transition_title_height,
        max(14, int(round(68 * scale))),
        max(10, int(round(40 * scale))),
        (215, 248, 255),
        max_lines=2,
        bold=False,
    )
    mashup_caption = render_text_layer(
        MASHUP_CAPTION,
        mashup_caption_width,
        mashup_caption_height,
        max(13, int(round(54 * scale))),
        max(9, int(round(34 * scale))),
        (255, 228, 137),
        max_lines=1,
        bold=True,
        pill=True,
    )
    mashup_title_a = render_text_layer(
        title_a,
        mashup_title_width,
        mashup_title_height,
        max(12, int(round(48 * scale))),
        max(9, int(round(31 * scale))),
        (250, 250, 252),
        max_lines=2,
        bold=False,
    )
    mashup_title_b = render_text_layer(
        title_b,
        mashup_title_width,
        mashup_title_height,
        max(12, int(round(48 * scale))),
        max(9, int(round(31 * scale))),
        (215, 248, 255),
        max_lines=2,
        bold=False,
    )
    mashup_x = render_text_layer(
        "X",
        x_layer_width,
        x_layer_height,
        max(16, int(round(78 * scale))),
        max(12, int(round(52 * scale))),
        (255, 232, 172),
        max_lines=1,
        bold=True,
    )
    bridge_width = (
        layout.mashup_card_b_x
        + layout.mashup_card_size
        - layout.mashup_card_a_x
    )
    bridge_height = max(26, int(round(layout.mashup_card_size * 0.25)))
    mashup_bridge_alpha = rounded_mask(
        bridge_width,
        bridge_height,
        max(4, bridge_height // 2),
    )
    mashup_bridge_alpha = np.clip(
        mashup_bridge_alpha.astype(np.float32) * 0.17,
        0,
        255,
    ).astype(np.uint8)

    return PairContext(
        pair_dir=pair_dir,
        width=width,
        height=height,
        title_a=title_a,
        title_b=title_b,
        song_dir_a=song_dir_a,
        song_dir_b=song_dir_b,
        cover_path_a=cover_path_a,
        cover_path_b=cover_path_b,
        background_a=background_a,
        background_b=background_b,
        transition_cards=transition_cards,
        mashup_cards=mashup_cards,
        transition_caption=transition_caption,
        transition_title_a=transition_title_a,
        transition_title_b=transition_title_b,
        mashup_caption=mashup_caption,
        mashup_title_a=mashup_title_a,
        mashup_title_b=mashup_title_b,
        mashup_x=mashup_x,
        mashup_bridge_alpha=mashup_bridge_alpha,
        layout=layout,
    )


def normalize_feature(values: np.ndarray, log_gain: float = 1.0) -> np.ndarray:
    """Normalize a nonnegative feature vector safely to the unit interval."""

    clean = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.maximum(clean, 0.0)
    if log_gain != 1.0:
        clean = np.log1p(clean * log_gain)
    maximum = float(clean.max(initial=0.0))
    if maximum <= 1e-12:
        return np.zeros_like(clean, dtype=np.float32)
    return np.clip(clean / maximum, 0.0, 1.0).astype(np.float32)


def smooth_feature(values: np.ndarray, coefficient: float) -> np.ndarray:
    """Apply vectorized one-pole smoothing without a Python frame loop."""

    source = np.asarray(values, dtype=np.float32)
    if source.size == 0:
        return source.copy()
    initial_state = np.array([coefficient * source[0]], dtype=np.float32)
    smoothed, _ = lfilter(
        [1.0 - coefficient],
        [1.0, -coefficient],
        source,
        zi=initial_state,
    )
    return np.asarray(smoothed, dtype=np.float32)


def analyze_audio_features(
    audio_path: Path,
    fps: int,
    waveform_width: int = WAVEFORM_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute bass, kick, and a 1D waveform envelope from one shared STFT."""

    if fps <= 0:
        raise ValueError("FPS must be positive")
    if not 128 <= waveform_width <= 256:
        raise ValueError("Waveform window size must be between 128 and 256 points")

    samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sample_rate <= 0 or samples.shape[0] == 0:
        raise ValueError(f"Audio is empty or invalid: {audio_path}")
    mono = np.mean(samples, axis=1, dtype=np.float32)
    duration = float(mono.size / sample_rate)

    n_fft = 2048
    hop_length = 512
    magnitude = np.abs(
        librosa.stft(mono, n_fft=n_fft, hop_length=hop_length, center=True)
    ).astype(np.float32)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    bass_rows = frequencies <= 180.0
    bass_spectrum = np.sqrt(np.mean(np.square(magnitude[bass_rows]), axis=0))
    energy = np.sqrt(np.mean(np.square(magnitude), axis=0))
    spectral_delta = np.diff(magnitude, axis=1, prepend=magnitude[:, :1])
    spectral_flux = np.sqrt(np.mean(np.square(np.maximum(spectral_delta, 0.0)), axis=0))

    bass_source = normalize_feature(bass_spectrum, log_gain=8.0)
    kick_source = normalize_feature(spectral_flux, log_gain=8.0)
    waveform = normalize_feature(energy, log_gain=12.0)

    total_frames = max(1, int(math.ceil(duration * fps)))
    source_times = np.arange(magnitude.shape[1], dtype=np.float64) * hop_length / sample_rate
    target_times = np.arange(total_frames, dtype=np.float64) / fps
    bass = np.interp(target_times, source_times, bass_source, left=0.0, right=float(bass_source[-1]))
    onset = np.interp(target_times, source_times, kick_source, left=0.0, right=0.0)
    bass = np.clip(smooth_feature(bass, BASS_SMOOTH), 0.0, 1.0)
    kick = np.clip((onset - 0.28) * 1.55, 0.0, 1.0).astype(np.float32)
    return bass.astype(np.float32), kick, waveform.astype(np.float32), duration


def waveform_window(
    waveform: np.ndarray,
    frame_index: int,
    total_frames: int,
    points: int = WAVEFORM_POINTS,
) -> np.ndarray:
    """Sample a zero-padded moving waveform window centered on playback."""

    if not 128 <= points <= 256:
        raise ValueError("Waveform window must contain 128 to 256 points")
    source = np.nan_to_num(np.asarray(waveform, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if source.size == 0 or total_frames <= 0:
        return np.zeros(points, dtype=np.float32)
    progress = float(np.clip(frame_index / max(1, total_frames - 1), 0.0, 1.0))
    center = progress * max(0, source.size - 1)
    offsets = np.arange(points, dtype=np.float32) - (points - 1) / 2.0
    indices = center + offsets
    sampled = np.interp(indices, np.arange(source.size), source, left=0.0, right=0.0)
    return np.clip(np.nan_to_num(sampled), 0.0, 1.0).astype(np.float32)


def validate_render_duration(metadata: dict, decoded_duration: float) -> None:
    """Validate decoded audio duration against rendered metadata when present."""

    expected = finite_score(metadata.get("rendered_duration_sec"), default=math.nan)
    if not math.isfinite(expected) or expected <= 0.0:
        return
    tolerance = max(0.10, expected * 0.005)
    difference = abs(decoded_duration - expected)
    if difference > tolerance:
        raise ValueError(
            "Decoded audio duration does not match rendered_duration_sec: "
            f"decoded={decoded_duration:.3f}s metadata={expected:.3f}s "
            f"difference={difference:.3f}s"
        )


def metadata_number(metadata: dict, key: str) -> float:
    """Read a required finite metadata number with a clear error."""

    value = finite_score(metadata.get(key), default=math.nan)
    if not math.isfinite(value):
        raise ValueError(f"Metadata is missing finite {key}")
    return value


def compute_transition_window(metadata: dict, duration: float = 0.0) -> Tuple[float, float]:
    """Derive render-relative transition bounds solely from the three bar counts."""

    del duration
    outgoing_bars = metadata_number(metadata, "outgoing_solo_bars")
    crossfade_bars = metadata_number(metadata, "crossfade_bars")
    incoming_bars = metadata_number(metadata, "incoming_solo_bars")
    total_bars = outgoing_bars + crossfade_bars + incoming_bars
    if outgoing_bars < 0.0 or incoming_bars < 0.0 or crossfade_bars <= 0.0 or total_bars <= 0.0:
        raise ValueError("Transition bar counts must define a positive crossfade")
    start = outgoing_bars / total_bars
    end = (outgoing_bars + crossfade_bars) / total_bars
    return float(start), float(end)


def compute_mashup_handoff_window(metadata: dict, duration: float) -> Tuple[float, float]:
    """Normalize mixer handoff render seconds, with a bar-based lead-in fallback."""

    if duration <= 0.0:
        raise ValueError("Mashup duration must be positive")
    start_seconds = finite_score(metadata.get("vocal_handoff_start_render_sec"), default=math.nan)
    end_seconds = finite_score(metadata.get("vocal_handoff_end_render_sec"), default=math.nan)
    if (
        math.isfinite(start_seconds)
        and math.isfinite(end_seconds)
        and 0.0 <= start_seconds < end_seconds
    ):
        start = float(np.clip(start_seconds / duration, 0.0, 1.0))
        end = float(np.clip(end_seconds / duration, start, 1.0))
        if end > start:
            return start, end

    lead_in_bars = metadata_number(metadata, "lead_in_bars")
    overlap_bars = finite_score(metadata.get("locked_overlap_bars"), default=math.nan)
    if not math.isfinite(overlap_bars) or overlap_bars <= 0.0:
        overlap_bars = finite_score(metadata.get("outgoing_phrase_bars"), default=math.nan)
    if lead_in_bars < 0.0 or not math.isfinite(overlap_bars) or overlap_bars <= 0.0:
        raise ValueError("Mixer metadata cannot derive a lead-in handoff window")
    total_bars = lead_in_bars + overlap_bars
    if lead_in_bars >= 1.0:
        start = (lead_in_bars - 1.0) / total_bars
        end = lead_in_bars / total_bars
    else:
        start = 0.0
        end = min(1.0, 1.0 / total_bars)
    return float(start), float(end)


def compute_mixer_handoff_window(metadata: dict, duration: float) -> Tuple[float, float]:
    """Expose the mashup handoff timing under the metadata mode name."""

    return compute_mashup_handoff_window(metadata, duration)


def smoothstep(value: float) -> float:
    """Clamp and ease a scalar through a cubic smoothstep curve."""

    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def progress_between(value: float, start: float, end: float) -> float:
    """Return eased progress within a normalized timing window."""

    return smoothstep((value - start) / max(1e-9, end - start))


def alpha_composite(
    base: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    mask: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """Alpha-composite only the intersecting ROI directly into the base array."""

    if opacity <= 0.0:
        return base
    overlay_h, overlay_w = overlay.shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(base.shape[1], x + overlay_w)
    bottom = min(base.shape[0], y + overlay_h)
    if left >= right or top >= bottom:
        return base

    overlay_left = left - x
    overlay_top = top - y
    overlay_right = overlay_left + right - left
    overlay_bottom = overlay_top + bottom - top
    roi = base[top:bottom, left:right]
    source = overlay[overlay_top:overlay_bottom, overlay_left:overlay_right]
    alpha_values = mask[overlay_top:overlay_bottom, overlay_left:overlay_right]
    if alpha_values.ndim == 3:
        alpha_values = alpha_values[..., 0]
    alpha = (
        np.clip(alpha_values.astype(np.float32) * float(opacity), 0.0, 255.0) / 255.0
    )[..., None]
    blended = source.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    roi[:] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return base


def composite_text_layer(
    base: np.ndarray,
    layer: TextLayer,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> np.ndarray:
    """Composite a cached Pillow text layer into a frame ROI in place."""

    return alpha_composite(base, layer.rgb, x, y, layer.alpha, opacity)


def composite_card(
    frame: np.ndarray,
    card: CardAssets,
    image: np.ndarray,
    x: int,
    y: int,
    glow_opacity: float,
) -> np.ndarray:
    """Composite one rounded card with cached shadow, glow, and border effects."""

    effect_x = x - card.effect_pad
    effect_y = y - card.effect_pad
    alpha_composite(frame, card.shadow_rgb, effect_x, effect_y, card.shadow_alpha, 1.0)
    alpha_composite(
        frame,
        card.glow_rgb,
        effect_x,
        effect_y,
        card.glow_alpha,
        float(np.clip(glow_opacity, 0.0, 0.85)),
    )
    alpha_composite(frame, image, x, y, card.mask, 1.0)
    alpha_composite(
        frame,
        card.border_rgb,
        x,
        y,
        card.border_alpha,
        float(np.clip(0.35 + glow_opacity, 0.0, 0.9)),
    )
    return frame


def make_background(
    background_a: np.ndarray,
    background_b: np.ndarray,
    progress: float,
    bass: float,
) -> np.ndarray:
    """Blend cached backgrounds and add a restrained audio-reactive pulse."""

    eased = smoothstep(progress)
    frame = cv2.addWeighted(background_a, 1.0 - eased, background_b, eased, 0.0)
    gain = 1.0 + 0.035 * float(np.clip(bass, 0.0, 1.0))
    return cv2.convertScaleAbs(frame, alpha=gain, beta=0.0)


def bass_pulsed_cover(image: np.ndarray, bass: float) -> np.ndarray:
    """Zoom and brighten a cover around its center in response to bass."""

    bass_amount = float(np.clip(bass, 0.0, 1.0))
    if bass_amount <= 1e-4:
        return image
    height, width = image.shape[:2]
    zoom = 1.0 + PULSE_STRENGTH * bass_amount
    transform = np.array(
        [
            [zoom, 0.0, (1.0 - zoom) * (width - 1) / 2.0],
            [0.0, zoom, (1.0 - zoom) * (height - 1) / 2.0],
        ],
        dtype=np.float32,
    )
    pulsed = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.convertScaleAbs(pulsed, alpha=1.0 + 0.06 * bass_amount, beta=0.0)


def draw_waveform_roi(
    frame: np.ndarray,
    values: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    progress: float,
    bass: float,
    kick: float,
    clip_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw a centered symmetric glowing waveform inside one local ROI."""

    if width < 2 or height < 2:
        return frame
    amplitudes = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    xs = np.linspace(3, width - 4, amplitudes.size, dtype=np.float32)
    center_y = (height - 1) / 2.0
    scale = height * 0.42 * (0.68 + 0.22 * bass + 0.18 * kick)
    upper = np.column_stack((xs, center_y - amplitudes * scale)).astype(np.int32)
    lower = np.column_stack((xs, center_y + amplitudes * scale)).astype(np.int32)
    polygon = np.concatenate((upper, lower[::-1]), axis=0).reshape((-1, 1, 2))

    fill_mask = np.zeros((height, width), dtype=np.uint8)
    core_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [polygon], 150, lineType=cv2.LINE_AA)
    thickness = max(1, int(round(width / 230.0 + 1.5 * kick)))
    cv2.polylines(core_mask, [upper.reshape((-1, 1, 2))], False, 255, thickness, cv2.LINE_AA)
    cv2.polylines(core_mask, [lower.reshape((-1, 1, 2))], False, 255, thickness, cv2.LINE_AA)
    cv2.line(
        core_mask,
        (3, int(round(center_y))),
        (width - 4, int(round(center_y))),
        68,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    if clip_mask is not None:
        local_clip = cv2.resize(clip_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        fill_mask = cv2.bitwise_and(fill_mask, local_clip)
        core_mask = cv2.bitwise_and(core_mask, local_clip)

    glow_mask = cv2.GaussianBlur(core_mask, (0, 0), max(2.0, height * 0.055))
    warm = np.array([255.0, 229.0, 166.0], dtype=np.float32)
    cool = np.array([112.0, 229.0, 255.0], dtype=np.float32)
    color = np.clip((1.0 - progress) * warm + progress * cool, 0, 255).astype(np.uint8)
    color_image = np.empty((height, width, 3), dtype=np.uint8)
    color_image[:] = color
    dark_image = np.zeros_like(color_image)
    alpha_composite(frame, dark_image, x, y, fill_mask, 0.35)
    alpha_composite(frame, color_image, x, y, glow_mask, 0.72)
    alpha_composite(frame, color_image, x, y, core_mask, 0.95)
    return frame


def draw_waveform_overlay_on_card(
    frame: np.ndarray,
    waveform: np.ndarray,
    frame_index: int,
    total_frames: int,
    progress: float,
    bass: float,
    kick: float,
    card_x: int,
    card_y: int,
    card_w: int,
    card_h: int,
    card_mask: np.ndarray,
) -> np.ndarray:
    """Overlay a moving local waveform on the lower area of one cover card."""

    band_height = max(24, int(round(card_h * 0.27)))
    relative_y = int(round(card_h * 0.67 - band_height / 2.0))
    local_mask = card_mask[relative_y : relative_y + band_height, :]
    values = waveform_window(waveform, frame_index, total_frames, WAVEFORM_POINTS)
    return draw_waveform_roi(
        frame,
        values,
        card_x,
        card_y + relative_y,
        card_w,
        band_height,
        progress,
        bass,
        kick,
        local_mask,
    )


def draw_waveform_bridge(
    frame: np.ndarray,
    waveform: np.ndarray,
    frame_index: int,
    total_frames: int,
    progress: float,
    bass: float,
    kick: float,
    left: int,
    right: int,
    center_y: int,
    height: int,
    bridge_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw a shared translucent waveform bridge across both mashup cards."""

    width = max(2, right - left)
    top = center_y - height // 2
    if bridge_alpha is None:
        bridge_alpha = rounded_mask(width, height, max(4, height // 2))
        bridge_alpha = np.clip(
            bridge_alpha.astype(np.float32) * 0.17,
            0,
            255,
        ).astype(np.uint8)
    elif bridge_alpha.shape != (height, width):
        bridge_alpha = cv2.resize(bridge_alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    bridge_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    alpha_composite(frame, bridge_rgb, left, top, bridge_alpha, 1.0)
    values = waveform_window(waveform, frame_index, total_frames, WAVEFORM_POINTS)
    return draw_waveform_roi(
        frame,
        values,
        left,
        top,
        width,
        height,
        progress,
        bass,
        kick,
    )


def draw_handoff_progress(
    frame: np.ndarray,
    left_center: int,
    right_center: int,
    y: int,
    progress: float,
    scale: float,
) -> np.ndarray:
    """Draw a compact mixer handoff rail timed by render-relative metadata."""

    progress = float(np.clip(progress, 0.0, 1.0))
    thickness = max(2, int(round(5 * scale)))
    cv2.line(
        frame,
        (left_center, y),
        (right_center, y),
        (32, 37, 52),
        thickness + max(2, thickness),
        cv2.LINE_AA,
    )
    end_x = int(round(left_center + (right_center - left_center) * progress))
    cv2.line(
        frame,
        (left_center, y),
        (end_x, y),
        (119, 227, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (end_x, y),
        max(3, int(round(8 * scale))),
        (255, 234, 178),
        -1,
        cv2.LINE_AA,
    )
    return frame


def transition_frames(
    context: PairContext,
    metadata: dict,
    bass: np.ndarray,
    kick: np.ndarray,
    waveform: np.ndarray,
    duration: float,
    fps: int,
) -> Iterator[np.ndarray]:
    """Yield audio-reactive transition frames using metadata bar timing."""

    del duration
    total_frames = len(bass)
    transition_start, transition_end = compute_transition_window(metadata)
    layout = context.layout
    card = context.transition_cards
    for frame_index in range(total_frames):
        normalized_time = frame_index / max(1, total_frames - 1)
        transition_progress = progress_between(normalized_time, transition_start, transition_end)
        bass_value = float(bass[frame_index])
        kick_value = float(kick[frame_index])
        seconds = frame_index / fps
        frame = make_background(
            context.background_a,
            context.background_b,
            transition_progress,
            bass_value,
        )

        mixed_cover = cv2.addWeighted(
            card.cover_a,
            1.0 - transition_progress,
            card.cover_b,
            transition_progress,
            0.0,
        )
        mixed_cover = bass_pulsed_cover(mixed_cover, bass_value)
        shake_x = int(round(math.sin(seconds * 28.0) * SHAKE_STRENGTH * kick_value))
        shake_y = int(round(math.cos(seconds * 23.0) * SHAKE_STRENGTH * 0.55 * kick_value))
        card_x = layout.transition_card_x + shake_x
        card_y = layout.transition_card_y + shake_y
        glow = 0.08 + 0.68 * bass_value + 0.08 * kick_value
        composite_card(frame, card, mixed_cover, card_x, card_y, glow)
        draw_waveform_overlay_on_card(
            frame,
            waveform,
            frame_index,
            total_frames,
            transition_progress,
            bass_value,
            kick_value,
            card_x,
            card_y,
            layout.transition_card_size,
            layout.transition_card_size,
            card.mask,
        )
        composite_text_layer(
            frame,
            context.transition_caption,
            layout.transition_caption_x,
            layout.transition_caption_y,
        )
        outgoing_title_opacity = 1.0 - smoothstep(transition_progress / 0.48)
        incoming_title_opacity = smoothstep((transition_progress - 0.52) / 0.48)
        composite_text_layer(
            frame,
            context.transition_title_a,
            layout.transition_title_x,
            layout.transition_title_y,
            outgoing_title_opacity,
        )
        composite_text_layer(
            frame,
            context.transition_title_b,
            layout.transition_title_x,
            layout.transition_title_y,
            incoming_title_opacity,
        )
        flash = 0.025 * kick_value * math.sin(math.pi * transition_progress)
        if flash > 0.0:
            cv2.addWeighted(frame, 1.0 - flash, frame, 0.0, 255.0 * flash, dst=frame)
        yield np.ascontiguousarray(frame, dtype=np.uint8)


def mashup_frames(
    context: PairContext,
    metadata: dict,
    bass: np.ndarray,
    kick: np.ndarray,
    waveform: np.ndarray,
    duration: float,
    fps: int,
) -> Iterator[np.ndarray]:
    """Yield dual-cover mashup frames timed by the mixer vocal handoff."""

    total_frames = len(bass)
    handoff_start, handoff_end = compute_mashup_handoff_window(metadata, duration)
    layout = context.layout
    card = context.mashup_cards
    scale = min(context.width / VIDEO_W, context.height / VIDEO_H)
    for frame_index in range(total_frames):
        normalized_time = frame_index / max(1, total_frames - 1)
        handoff_progress = progress_between(normalized_time, handoff_start, handoff_end)
        bass_value = float(bass[frame_index])
        kick_value = float(kick[frame_index])
        seconds = frame_index / fps
        background_progress = 0.34 + 0.32 * handoff_progress
        frame = make_background(
            context.background_a,
            context.background_b,
            background_progress,
            bass_value,
        )

        shake_x = int(round(math.sin(seconds * 28.0) * SHAKE_STRENGTH * 0.55 * kick_value))
        shake_y = int(round(math.cos(seconds * 23.0) * SHAKE_STRENGTH * 0.35 * kick_value))
        card_a_x = layout.mashup_card_a_x + shake_x
        card_b_x = layout.mashup_card_b_x - shake_x
        card_a_y = layout.mashup_card_y + shake_y
        card_b_y = layout.mashup_card_y - shake_y
        cover_a = bass_pulsed_cover(card.cover_a, bass_value)
        cover_b = bass_pulsed_cover(card.cover_b, bass_value)
        glow = 0.08 + 0.62 * bass_value + 0.06 * kick_value
        composite_card(
            frame,
            card,
            cover_a,
            card_a_x,
            card_a_y,
            glow + 0.08 * (1.0 - handoff_progress),
        )
        composite_card(
            frame,
            card,
            cover_b,
            card_b_x,
            card_b_y,
            glow + 0.08 * handoff_progress,
        )

        bridge_left = layout.mashup_card_a_x
        bridge_right = layout.mashup_card_b_x + layout.mashup_card_size
        bridge_center_y = layout.mashup_card_y + int(round(layout.mashup_card_size * 0.69))
        bridge_height = max(26, int(round(layout.mashup_card_size * 0.25)))
        draw_waveform_bridge(
            frame,
            waveform,
            frame_index,
            total_frames,
            handoff_progress,
            bass_value,
            kick_value,
            bridge_left,
            bridge_right,
            bridge_center_y,
            bridge_height,
            context.mashup_bridge_alpha,
        )
        rail_y = layout.mashup_card_y - max(8, int(round(30 * scale)))
        draw_handoff_progress(
            frame,
            card_a_x + layout.mashup_card_size // 2,
            card_b_x + layout.mashup_card_size // 2,
            rail_y,
            handoff_progress,
            scale,
        )
        composite_text_layer(
            frame,
            context.mashup_caption,
            layout.mashup_caption_x,
            layout.mashup_caption_y,
        )
        composite_text_layer(
            frame,
            context.mashup_title_a,
            layout.mashup_title_a_x,
            layout.mashup_title_y,
        )
        composite_text_layer(
            frame,
            context.mashup_title_b,
            layout.mashup_title_b_x,
            layout.mashup_title_y,
        )
        composite_text_layer(
            frame,
            context.mashup_x,
            layout.mashup_x_x,
            layout.mashup_x_y,
        )
        flash = 0.018 * kick_value
        if flash > 0.0:
            cv2.addWeighted(frame, 1.0 - flash, frame, 0.0, 255.0 * flash, dst=frame)
        yield np.ascontiguousarray(frame, dtype=np.uint8)


def ffmpeg_rawvideo_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> subprocess.Popen:
    """Open an H.264 yuv420p ffmpeg process that accepts raw RGB frames."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required on PATH to encode video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def ffmpeg_mux_audio(silent_video: Path, audio_path: Path, output_path: Path) -> None:
    """Mux AAC audio and atomically replace the final H.264 video."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required on PATH to mux audio")
    if not silent_video.is_file() or silent_video.stat().st_size == 0:
        raise RuntimeError(f"Silent video is missing or empty: {silent_video}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"Render audio is missing or empty: {audio_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    muxed_video = output_path.with_name(
        f".{output_path.stem}.mux.{uuid.uuid4().hex}.mp4"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        str(muxed_video),
    ]
    try:
        subprocess.run(command, check=True)
        if not muxed_video.is_file() or muxed_video.stat().st_size == 0:
            raise RuntimeError(f"Muxed video is missing or empty: {muxed_video}")
        muxed_video.replace(output_path)
    finally:
        try:
            muxed_video.unlink(missing_ok=True)
        except OSError:
            pass


def encode_frames_with_audio(
    frames: Iterable[np.ndarray],
    audio_path: Path,
    output_path: Path,
    mode_name: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Encode generated frames and always remove the unique mode-named temp file."""

    unique = uuid.uuid4().hex
    silent_video = output_path.with_name(
        f".{output_path.stem}.{mode_name}.{unique}.silent.mp4"
    )
    process: Optional[subprocess.Popen] = None
    written_frames = 0
    try:
        process = ffmpeg_rawvideo_writer(silent_video, width, height, fps)
        if process.stdin is None:
            raise RuntimeError("Could not open ffmpeg raw-video stdin")
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError(
                    f"Invalid frame shape or dtype: {frame.shape} {frame.dtype}"
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            written_frames += 1
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg video encoder failed with exit code {return_code}")
        if written_frames == 0:
            raise RuntimeError("No video frames were generated")
        ffmpeg_mux_audio(silent_video, audio_path, output_path)
    finally:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        try:
            silent_video.unlink(missing_ok=True)
        except OSError:
            pass


def render_transition_video(
    asset: RenderAsset,
    context: PairContext,
    output_path: Path,
    fps: int = FPS,
) -> None:
    """Analyze and render one explicit transition asset to its final MP4."""

    validate_render_duration(asset.metadata, float(sf.info(str(asset.wav_path)).duration))
    bass, kick, waveform, duration = analyze_audio_features(asset.wav_path, fps)
    validate_render_duration(asset.metadata, duration)
    frames = transition_frames(
        context,
        asset.metadata,
        bass,
        kick,
        waveform,
        duration,
        fps,
    )
    encode_frames_with_audio(
        frames,
        asset.wav_path,
        output_path,
        "transition",
        context.width,
        context.height,
        fps,
    )


def render_mashup_video(
    asset: RenderAsset,
    context: PairContext,
    output_path: Path,
    fps: int = FPS,
) -> None:
    """Analyze and render one explicit mixer asset as the mashup MP4."""

    validate_render_duration(asset.metadata, float(sf.info(str(asset.wav_path)).duration))
    bass, kick, waveform, duration = analyze_audio_features(asset.wav_path, fps)
    validate_render_duration(asset.metadata, duration)
    frames = mashup_frames(
        context,
        asset.metadata,
        bass,
        kick,
        waveform,
        duration,
        fps,
    )
    encode_frames_with_audio(
        frames,
        asset.wav_path,
        output_path,
        "mashup",
        context.width,
        context.height,
        fps,
    )


def requested_output_modes(mode: str) -> Tuple[str, ...]:
    """Expand the CLI mode into ordered output mode names."""

    if mode == "both":
        return "transition", "mashup"
    if mode in {"transition", "mashup"}:
        return (mode,)
    raise ValueError(f"Unsupported output mode: {mode}")


def summarize_pair_status(mode_results: Mapping[str, Mapping[str, str]]) -> str:
    """Summarize per-mode outcomes as ok, partial, skip, or error."""

    statuses = [result.get("status", "error") for result in mode_results.values()]
    successes = sum(status in {"ok", "ready"} for status in statuses)
    if successes == len(statuses) and statuses:
        return "ok"
    if successes:
        return "partial"
    if statuses and all(status == "skip" for status in statuses):
        return "skip"
    return "error"


def process_pair_folder(
    pair_dir: Path,
    song_dirs: Sequence[Path],
    output_root: Path = OUTPUT_ROOT,
    mode: str = "both",
    width: int = VIDEO_W,
    height: int = VIDEO_H,
    fps: int = FPS,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Process one pair folder with explicit independent mode outcomes."""

    if not pair_dir.is_dir():
        return {
            "status": "skip",
            "pair_dir": str(pair_dir),
            "modes": {},
            "reason": "render pair folder does not exist",
        }
    validate_video_settings(width, height, fps)
    selected = discover_mode_assets(pair_dir)
    output_modes = requested_output_modes(mode)
    mode_results: Dict[str, Dict[str, str]] = {}
    available_modes: List[str] = []
    for output_mode in output_modes:
        asset_mode = "transition" if output_mode == "transition" else "mixer"
        asset = selected[asset_mode]
        if asset is None:
            mode_results[output_mode] = {
                "status": "skip",
                "reason": f"missing explicit metadata mode '{asset_mode}' asset",
            }
        else:
            available_modes.append(output_mode)

    context: Optional[PairContext] = None
    if available_modes:
        try:
            requested_assets = {
                "transition" if output_mode == "transition" else "mixer": selected[
                    "transition" if output_mode == "transition" else "mixer"
                ]
                for output_mode in available_modes
            }
            context = build_pair_context(
                pair_dir,
                requested_assets,
                song_dirs,
                width,
                height,
            )
        except Exception as error:
            for output_mode in available_modes:
                mode_results[output_mode] = {
                    "status": "error",
                    "reason": f"pair context failed: {error}",
                }

    if context is not None:
        output_dir = output_root / pair_dir.name
        for output_mode in available_modes:
            asset_mode = "transition" if output_mode == "transition" else "mixer"
            asset = selected[asset_mode]
            if asset is None:
                continue
            output_path = output_dir / f"{output_mode}.mp4"
            if dry_run:
                mode_results[output_mode] = {
                    "status": "ready",
                    "asset_mode": asset_mode,
                    "wav": str(asset.wav_path),
                    "metadata": str(asset.metadata_path),
                    "output": str(output_path),
                    "title_a": context.title_a,
                    "title_b": context.title_b,
                }
                continue
            try:
                if output_mode == "transition":
                    render_transition_video(asset, context, output_path, fps)
                else:
                    render_mashup_video(asset, context, output_path, fps)
                mode_results[output_mode] = {
                    "status": "ok",
                    "asset_mode": asset_mode,
                    "wav": str(asset.wav_path),
                    "metadata": str(asset.metadata_path),
                    "output": str(output_path),
                    "title_a": context.title_a,
                    "title_b": context.title_b,
                }
            except Exception as error:
                mode_results[output_mode] = {
                    "status": "error",
                    "asset_mode": asset_mode,
                    "reason": str(error),
                }

    return {
        "status": summarize_pair_status(mode_results),
        "pair_dir": str(pair_dir),
        "modes": mode_results,
    }


def iter_render_folders(root: Path) -> List[Path]:
    """List immediate non-symlink render folders in deterministic order."""

    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    folders: List[Path] = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            if path.resolve().parent == resolved_root:
                folders.append(path)
        except OSError:
            continue
    return sorted(
        folders,
        key=lambda path: (path.name.casefold(), path.name),
    )


def iter_transition_dirs(root: Path) -> List[Path]:
    """Return immediate render folders under the legacy function name."""

    return iter_render_folders(root)


def resolve_render_folder_title(title: str, renders_root: Path = RENDERS_ROOT) -> Path:
    """Resolve one immediate folder title and reject traversal or outside paths."""

    raw = Path(title)
    if raw.is_absolute() or len(raw.parts) != 1 or raw.name in {"", ".", ".."}:
        raise ValueError("Render folder must be one immediate folder title, not a path")
    root = renders_root.resolve()
    candidate = (root / raw.name).resolve()
    if candidate.parent != root:
        raise ValueError("Render folder resolves outside the renders root")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Render folder not found: {candidate}")
    return candidate


def validate_video_settings(width: int, height: int, fps: int) -> None:
    """Validate bounded portrait yuv420p dimensions and frame rate."""

    if width < 160 or height < 284:
        raise ValueError("Video dimensions must be at least 160x284")
    if width % 2 or height % 2:
        raise ValueError("Video width and height must be even for yuv420p")
    aspect_ratio = width / height
    if not 0.5 <= aspect_ratio <= 0.65:
        raise ValueError("Video dimensions must use a portrait aspect ratio near 9:16")
    if width * height > 2160 * 3840:
        raise ValueError("Video dimensions must not exceed 2160x3840 total pixels")
    if not 1 <= fps <= 60:
        raise ValueError("FPS must be between 1 and 60")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for pair discovery and rendering."""

    parser = argparse.ArgumentParser(
        description=(
            "Render one transition.mp4 and/or mashup.mp4 from explicit-mode "
            "assets in immediate renders folders."
        )
    )
    parser.add_argument(
        "--render-folder-title",
        "--pair-dir",
        dest="render_folder_title",
        help="Exact immediate folder title under PROJECT_ROOT/renders.",
    )
    parser.add_argument(
        "--transition-dir",
        help="Deprecated alias for --render-folder-title.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "transition", "mashup"),
        default="both",
        help="Output mode to render (default: both).",
    )
    parser.add_argument("--width", type=int, default=VIDEO_W, help="Video width (default: 1080).")
    parser.add_argument("--height", type=int, default=VIDEO_H, help="Video height (default: 1920).")
    parser.add_argument("--fps", type=int, default=FPS, help="Frame rate (default: 24).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery, songs, covers, titles, and planned outputs without rendering.",
    )
    return parser


def print_pair_result(result: Mapping[str, object]) -> None:
    """Print concise per-mode pair outcomes for CLI users."""

    pair_name = Path(str(result.get("pair_dir", "unknown"))).name
    mode_results = result.get("modes", {})
    if not isinstance(mode_results, Mapping):
        print(f"[skip] {pair_name}: {result.get('reason', 'unknown error')}")
        return
    for mode_name, raw_mode_result in mode_results.items():
        if not isinstance(raw_mode_result, Mapping):
            continue
        status = raw_mode_result.get("status", "error")
        if status in {"ok", "ready"}:
            verb = "ready" if status == "ready" else "ok"
            print(
                f"[{verb}] {pair_name} {mode_name}: "
                f"{raw_mode_result.get('output', '')} "
                f"({raw_mode_result.get('title_a', '')} -> {raw_mode_result.get('title_b', '')})"
            )
        else:
            print(
                f"[{status}] {pair_name} {mode_name}: "
                f"{raw_mode_result.get('reason', 'unknown error')}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CLI discovery and render requested immediate pair folders."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.render_folder_title is not None and args.transition_dir is not None:
        parser.error("Use only one folder option")
    folder_title = (
        args.render_folder_title
        if args.render_folder_title is not None
        else args.transition_dir
    )
    if args.transition_dir is not None:
        print(
            "warning: --transition-dir is deprecated; use --render-folder-title",
            file=sys.stderr,
        )
    try:
        validate_video_settings(args.width, args.height, args.fps)
    except ValueError as error:
        parser.error(str(error))

    if not SONGS_ROOT.is_dir():
        raise FileNotFoundError(f"Songs root not found: {SONGS_ROOT}")
    if not RENDERS_ROOT.is_dir():
        raise FileNotFoundError(f"Renders root not found: {RENDERS_ROOT}")
    song_dirs = discover_song_folders(SONGS_ROOT)

    if folder_title is not None:
        try:
            pair_dirs = [resolve_render_folder_title(folder_title)]
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    else:
        pair_dirs = iter_render_folders(RENDERS_ROOT)
    if not pair_dirs:
        print("No immediate render folders found.")
        return 0

    results: List[Dict[str, object]] = []
    for pair_dir in pair_dirs:
        result = process_pair_folder(
            pair_dir=pair_dir,
            song_dirs=song_dirs,
            output_root=OUTPUT_ROOT,
            mode=args.mode,
            width=args.width,
            height=args.height,
            fps=args.fps,
            dry_run=args.dry_run,
        )
        results.append(result)
        print_pair_result(result)

    video_count = sum(
        1
        for result in results
        for mode_result in (
            result.get("modes", {}).values()
            if isinstance(result.get("modes"), Mapping)
            else []
        )
        if isinstance(mode_result, Mapping) and mode_result.get("status") == "ok"
    )
    if not args.dry_run:
        print(f"Rendered {video_count} video(s) under {OUTPUT_ROOT}")
    return 0 if all(result.get("status") == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
