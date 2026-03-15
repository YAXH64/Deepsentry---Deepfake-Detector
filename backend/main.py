"""
main.py
FastAPI backend for the AI Deepfake Detector.

Routes:
  GET  /              → health check
  POST /detect/image  → analyse an image
  POST /detect/video  → analyse a video

Run:
  python main.py
  → http://localhost:8000
  → http://localhost:8000/docs   (auto-generated Swagger UI)
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_model, predict_image, predict_video
from processor import extract_frames, preprocess_image


# ── Startup: load model before first request ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading model …")
    load_model()
    print("[startup] Ready.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Deepfake Detector",
    description="Detects whether an image or video is Real or Deepfake using a pretrained ViT model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # covers localhost, LAN, and deployed origins
    allow_origin_regex=".*",    # also covers file:// (null origin) for local HTML files
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── Response model ────────────────────────────────────────────────────────────

class Result(BaseModel):
    label:      str    # "Real" or "Deepfake"
    confidence: float  # 0.0 – 100.0
    media_type: str    # "image" or "video"
    elapsed_ms: float  # total processing time


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/detect/image", response_model=Result)
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image (jpg / png / webp / bmp) and get a prediction.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    t0      = time.perf_counter()
    img     = await preprocess_image(file)
    label, confidence = predict_image(img)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return Result(label=label, confidence=confidence, media_type="image", elapsed_ms=elapsed)


@app.post("/detect/video", response_model=Result)
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video (mp4 / avi / mov / mkv / webm) and get a prediction.
    Up to 20 frames are sampled and averaged for the final result.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    t0      = time.perf_counter()
    frames  = await extract_frames(file)
    label, confidence = predict_video(frames)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    return Result(label=label, confidence=confidence, media_type="video", elapsed_ms=elapsed)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)