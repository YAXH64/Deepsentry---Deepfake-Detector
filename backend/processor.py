"""
processor.py
Converts uploaded files into PIL Images for the model.

preprocess_image(file)  →  PIL.Image (single image)
extract_frames(file)    →  list[PIL.Image]  (up to MAX_FRAMES from a video)

Both functions are async — called directly from FastAPI route handlers.
"""

import os
import tempfile

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_FRAMES = 20  # frames sampled per video — more = better accuracy, slower

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _ext(filename: str) -> str:
    return os.path.splitext(filename)[-1].lower()


def _check_ext(filename: str, allowed: set):
    ext = _ext(filename)
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}",
        )


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


async def preprocess_image(file: UploadFile) -> Image.Image:
    """
    Read an uploaded image and return an RGB PIL Image.
    The HuggingFace ViTImageProcessor handles all resizing/normalisation.
    """
    _check_ext(file.filename, ALLOWED_IMAGE)

    raw    = await file.read()
    np_arr = np.frombuffer(raw, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image — file may be corrupted.")

    return _bgr_to_pil(frame)


async def extract_frames(file: UploadFile) -> list:
    """
    Read an uploaded video, sample up to MAX_FRAMES evenly across its
    full duration, and return them as a list of RGB PIL Images.

    OpenCV needs a real file path for video decoding, so we write to a
    temp file and delete it when done.
    """
    _check_ext(file.filename, ALLOWED_VIDEO)

    raw    = await file.read()
    suffix = _ext(file.filename)

    # Write to temp file — OpenCV cannot decode video from a memory buffer
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            raise HTTPException(status_code=400, detail="Video has no frames.")

        # Evenly-spaced indices across the whole video
        n_samples = min(MAX_FRAMES, total)
        indices   = np.linspace(0, total - 1, n_samples, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(_bgr_to_pil(frame))

        cap.release()

    finally:
        os.unlink(tmp.name)  # always clean up

    if not frames:
        raise HTTPException(status_code=400, detail="Failed to extract frames from video.")

    return frames