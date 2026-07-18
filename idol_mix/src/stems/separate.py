import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = Path(os.environ.get("KPOP_DJ_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_NAME = "htdemucs"
EXPECTED_STEMS = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]


def find_song_wavs(data_dir: Path) -> list[Path]:
    wavs = []
    for song_folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        candidates = sorted(song_folder.glob("*.wav"))
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Multiple top-level WAVs in {song_folder}: {names}")
        if candidates:
            wavs.append(candidates[0])
    return wavs


def stems_output_dir(song_dir: Path) -> Path:
    return song_dir / "stems"


def stems_done(song_dir: Path, source_path: Path | None = None) -> bool:
    stems_dir = stems_output_dir(song_dir)
    stem_paths = [stems_dir / name for name in EXPECTED_STEMS]
    if not stems_dir.exists() or not all(path.is_file() for path in stem_paths):
        return False
    if source_path is not None:
        source_mtime = source_path.stat().st_mtime_ns
        if any(path.stat().st_mtime_ns < source_mtime for path in stem_paths):
            return False
    return True


def run_demucs_for_file(src_path: Path):
    song_dir = src_path.parent
    track_id = src_path.stem

    if not src_path.is_file():
        print(f"[skip:not-a-file] {src_path}")
        return

    if stems_done(song_dir, src_path):
        print(f"[skip:already-separated] {song_dir.name}")
        return

    temp_out = song_dir / "_demucs_tmp"
    temp_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", MODEL_NAME,
        "-d", "cpu",
        str(src_path),
        "-o", str(temp_out),
    ]

    print(f"[run] {song_dir.name} :: {track_id}")
    subprocess.run(cmd, check=True)

    demucs_track_dir = temp_out / MODEL_NAME / src_path.stem
    if not demucs_track_dir.exists():
        raise RuntimeError(f"Demucs output folder not found: {demucs_track_dir}")

    final_stems_dir = stems_output_dir(song_dir)
    final_stems_dir.mkdir(parents=True, exist_ok=True)

    for stem_name in EXPECTED_STEMS:
        src_stem = demucs_track_dir / stem_name
        dst_stem = final_stems_dir / stem_name
        if not src_stem.is_file():
            raise RuntimeError(f"Missing expected stem: {src_stem}")
        src_stem.replace(dst_stem)

    for leftover in sorted(demucs_track_dir.glob("*")):
        if leftover.is_file():
            leftover.unlink()

    if demucs_track_dir.exists():
        demucs_track_dir.rmdir()
    model_dir = temp_out / MODEL_NAME
    if model_dir.exists() and not any(model_dir.iterdir()):
        model_dir.rmdir()
    if temp_out.exists() and not any(temp_out.iterdir()):
        temp_out.rmdir()

    if not all((final_stems_dir / name).is_file() for name in EXPECTED_STEMS):
        raise RuntimeError(f"Demucs finished but expected stems not found for {song_dir.name}")

    print(f"[done] {song_dir.name}")



def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate all unprocessed songs with Demucs")
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

    files = find_song_wavs(data_dir)
    if not files:
        print(f"No WAV files found under song folders in {data_dir}")
        return

    total = 0
    failures = []
    for src_path in files:
        total += 1
        try:
            run_demucs_for_file(src_path)
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run later to continue from unfinished files.")
            raise
        except Exception as e:
            print(f"[error] {src_path}: {e}")
            failures.append((src_path, e))

    print(f"\nScanned wav files: {total}")
    if failures:
        raise RuntimeError(f"Stem separation failed for {len(failures)} song(s)")


if __name__ == "__main__":
    main()
