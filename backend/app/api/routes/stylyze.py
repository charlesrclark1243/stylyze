import logging
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Resolved from this file rather than the working directory, so the app finds the checkpoint however it's launched.
# MODEL_CHECKPOINT overrides it, e.g. for weights mounted into the container
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "model.onnx"
CHECKPOINT_PATH = Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))


def load_model(checkpoint_path: Path) -> ort.InferenceSession | None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(
        os.environ.get("OMP_NUM_THREADS") or len(os.sched_getaffinity(0))
    )
    options.enable_cpu_mem_arena = False

    try:
        session = ort.InferenceSession(
            str(checkpoint_path), options, providers=["CPUExecutionProvider"]
        )
        return session
    except ort.capi.onnxruntime_pybind11_state.NoSuchFile:
        logger.error("Error loading ONNX model: File not found.")
        return None
    except ort.capi.onnxruntime_pybind11_state.RuntimeException:
        logger.error(
            "Error loading ONNX model: ONNX runtime exception. The model may be corrupted or incompatible."
        )
        return None
    except Exception:  # noqa: BLE001
        logger.error("Unexpected error loading ONNX model.")
        return None


session = load_model(CHECKPOINT_PATH)
semaphore = threading.Semaphore(1)  # Limit to one inference at a time

router = APIRouter()

ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_MEM_SIZE = 10 * 1024 * 1024  # 10 MB

MAX_PIXELS = 50 * 1_000_000  # 50 million pixels MAX (warning past 25 million)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS // 2  # Prevent decompression bomb DOS attacks


def load_image(file: UploadFile):
    """
    Load and validate an image file.

    Args:
        file (UploadFile): The image file to load.

    Returns:
        Image: The loaded and validated image.

    Raises:
        HTTPException: If the image is invalid or exceeds size limits.
    """

    # safeguards
    if file.content_type not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Invalid image format. Only JPEG and PNG are supported.",
        )

    if file.size is not None and file.size > MAX_MEM_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image size exceeds the maximum allowed size of {MAX_MEM_SIZE / (1024 * 1024)} MB.",
        )
    elif file.size is None:
        raise HTTPException(
            status_code=400,
            detail="Image size could not be determined. Please ensure the image is valid.",
        )

    # load and verify, then return if it passes
    try:
        image = ImageOps.exif_transpose(Image.open(BytesIO(file.file.read()))).convert(
            "RGB"
        )
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. The file could not be opened or is corrupted.",
        )

    return image


def resize_short_side(image: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Resize the image so that its shorter side is equal to target_size, maintaining the aspect ratio.

    Args:
        image (Image.Image): The input image to resize.
        target_size (int): The desired size for the shorter side of the image.

    Returns:
        Image.Image: The resized image.
    """

    w, h = image.size
    scale = target_size / min(w, h)

    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)


def to_array(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy array with shape (1, 3, H, W) and values in [0, 1].

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        np.ndarray: The image as a NumPy array with shape (1, 3, H, W).
    """

    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


@router.post("/stylyze")
def stylyze(
    content: Annotated[UploadFile, File(...)],
    style: Annotated[UploadFile, File(...)],
    alpha: Annotated[float, Form(ge=0.0, le=1.0)] = 1.0,
):
    """
    Perform style transfer on the uploaded content and style images using the pre-trained model. Returns the stylized image.

    Args:
        content (UploadFile): The content image file.
        style (UploadFile): The style image file.
        alpha (float): Style strength, from 0 (keep the content's look) to 1 (full style). Defaults to 1.

    Returns:
        Response: The stylized image in JPEG format.
    """

    if session is None:
        raise HTTPException(
            status_code=503, detail="The style transfer model is not loaded."
        )

    content_image = resize_short_side(load_image(content))
    style_image = resize_short_side(load_image(style))

    width, height = content_image.size
    content_image = content_image.crop((0, 0, width - width % 8, height - height % 8))

    content_image = to_array(content_image)
    style_image = to_array(style_image)

    if not semaphore.acquire(timeout=30):
        raise HTTPException(
            status_code=429,
            detail="The server is busy stylizing another image. Please try again shortly.",
            headers={
                "Retry-After": "30"
            },  # suggests the client wait 30 seconds before retrying
        )

    try:
        stylyzed = session.run(
            None,
            {
                "content": content_image,
                "style": style_image,
                "alpha": np.array(alpha, dtype=np.float32),
            },
        )[0]
    finally:
        semaphore.release()

    # Convert the output tensor to a PIL image
    stylyzed_image = Image.fromarray(
        (stylyzed[0].transpose(1, 2, 0).clip(0, 1) * 255).round().astype("uint8")
    )

    buffer = BytesIO()
    # Quality 90 keeps painterly textures clean; the default of 75 visibly blocks them
    stylyzed_image.save(buffer, format="JPEG", quality=90)

    return Response(content=buffer.getvalue(), media_type="image/jpeg")
