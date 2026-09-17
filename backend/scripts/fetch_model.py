"""Fetches the trained checkpoint from its GitHub release.

The weights are 63 MB and only change when the model is retrained, so they ship as a
release asset instead of living in git. Run this once after cloning; the Docker build
runs it too, so the image carries its own copy.

Set MODEL_URL and MODEL_SHA256 to pull different weights without editing this file.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    os.environ.get("MODEL_URL")
    or "https://github.com/charlesrclark1243/stylyze/releases/download/model-v1/model.pth"
)
MODEL_SHA256 = (
    os.environ.get("MODEL_SHA256")
    or "6eb56be5cd858ede07c94fb98dc4c58550585482dd0f51a774a6c5bb90e9217b"
).lower()

# Matches DEFAULT_CHECKPOINT in app/api/routes/stylyze.py, from the backend root
DESTINATION = Path(__file__).resolve().parents[1] / "checkpoints" / "model.pth"


def sha256(path: Path) -> str:
    """Hashes a file in 1 MB chunks, so the checkpoint never lands in memory twice."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    if DESTINATION.is_file() and sha256(DESTINATION) == MODEL_SHA256:
        print(f"{DESTINATION} is already up to date")
        return 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, DESTINATION)

    # Release assets can be replaced under the same tag, so verify rather than trust the URL
    digest = sha256(DESTINATION)
    if digest != MODEL_SHA256:
        DESTINATION.unlink()
        print(
            f"Checksum mismatch: expected {MODEL_SHA256}, got {digest}", file=sys.stderr
        )
        return 1

    print(f"Wrote {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
