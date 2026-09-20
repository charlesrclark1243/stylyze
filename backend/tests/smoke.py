"""Smoke tests for a running backend container.

Run against the image that is about to ship, not against the source tree - most of the
failures these catch (a wrong checkpoint path, a renamed global, a broken semaphore) only
appear once the app is actually serving.

    python backend/tests/smoke.py http://localhost:8000
"""

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
PRESETS = Path(__file__).resolve().parents[2] / "frontend" / "public" / "presets"
BOUNDARY = "----stylyzesmokeboundary"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f' - {detail}' if detail else ''}")
    if not condition:
        failures.append(name)


def post_stylyze(content: bytes, style: bytes, alpha: str = "1.0", content_type: str = "image/jpeg"):
    """Posts a multipart request and returns (status, content_type_header, body)."""
    body = b""
    for field, data, ctype in (("content", content, content_type), ("style", style, "image/jpeg")):
        body += (
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{field}.bin"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        body += data + b"\r\n"
    body += f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="alpha"\r\n\r\n{alpha}\r\n'.encode()
    body += f"--{BOUNDARY}--\r\n".encode()

    request = urllib.request.Request(
        f"{BASE}/api/stylyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def exif_rotated_jpeg(source: Path) -> bytes:
    """A phone-style photo: pixels stored rotated, with EXIF saying to rotate them back."""
    upright = Image.open(source).convert("RGB")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW for display

    buffer = io.BytesIO()
    upright.transpose(Image.Transpose.ROTATE_90).save(buffer, "JPEG", exif=exif)

    return buffer.getvalue()


def main() -> int:
    content_path, style_path = PRESETS / "great-wave.jpg", PRESETS / "the-scream.jpg"
    content, style = content_path.read_bytes(), style_path.read_bytes()

    print("health")
    with urllib.request.urlopen(f"{BASE}/health", timeout=30) as response:
        health = json.loads(response.read())
    check("status ok", health.get("status") == "ok", str(health))
    check("model is loaded", health.get("model_loaded") is True, str(health))

    print("stylyze")
    status, header, result = post_stylyze(content, style)
    check("returns 200", status == 200, f"got {status}")
    check("returns a jpeg", header.startswith("image/jpeg"), header)

    image = Image.open(io.BytesIO(result))
    width, height = image.size
    check("output decodes as an image", image.format == "JPEG", str(image.format))
    # The exported graph floors content dimensions to multiples of 8, so the API crops to match
    check("dimensions are multiples of 8", width % 8 == 0 and height % 8 == 0, f"{width}x{height}")

    source_width, source_height = Image.open(content_path).size
    check(
        "orientation is preserved",
        (width > height) == (source_width > source_height),
        f"{width}x{height} from {source_width}x{source_height}",
    )

    print("exif orientation")
    # A phone photo the browser shows upright must not come back rotated
    _, _, rotated = post_stylyze(exif_rotated_jpeg(content_path), style)
    rotated_width, rotated_height = Image.open(io.BytesIO(rotated)).size
    check(
        "exif rotation is applied",
        rotated_width > rotated_height,
        f"{rotated_width}x{rotated_height}, expected landscape",
    )

    print("alpha")
    # Guards against alpha being constant-folded into a future ONNX export, which no
    # other check would notice: the endpoint keeps working, the slider just stops doing anything
    _, _, full = post_stylyze(content, style, alpha="1.0")
    _, _, none = post_stylyze(content, style, alpha="0.0")
    check("alpha changes the output", full != none)

    print("rejections")
    status, _, _ = post_stylyze(b"this is not an image", style, content_type="text/plain")
    check("non-image is rejected with 415", status == 415, f"got {status}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
