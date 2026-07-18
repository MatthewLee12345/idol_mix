"""Run the canonical six-stage DJ build workflow without scheduling it."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    ("ingest", "src.ingest.ingest"),
    ("analyze", "src.analysis.analyze"),
    ("pairing", "src.pairing.pairing_identification"),
    ("stems", "src.stems.separate"),
    ("render", "src.rendering.render"),
    ("visualization", "src.visualization.visualizer"),
)


def selected_stages(start_at: str, stop_after: str):
    """Return the inclusive ordered stage slice requested by the caller."""
    names = [name for name, _ in STAGES]
    start = names.index(start_at)
    stop = names.index(stop_after)
    if start > stop:
        raise ValueError("--start-at must not come after --stop-after")
    return STAGES[start:stop + 1]


def main(argv=None):
    """Execute each requested stage with the active Python interpreter."""
    names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=names, default=names[0])
    parser.add_argument("--stop-after", choices=names, default=names[-1])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered module commands without running them",
    )
    args = parser.parse_args(argv)

    for name, module in selected_stages(args.start_at, args.stop_after):
        command = [sys.executable, "-m", module]
        print(f"[workflow:{name}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
