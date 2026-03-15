"""
model.py
Pretrained deepfake detector using:
  prithivMLmods/Deep-Fake-Detector-v2-Model
  Architecture : ViT (vit-base-patch16-224-in21k fine-tuned)
  Accuracy     : ~92%  (56,001-image test set)
  Labels       : "Realism" (real) | "Deepfake"

Weights download automatically on first run (~330 MB).
Cached forever in ~/.cache/huggingface/ — only downloaded once.
"""

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "Realism" is the model's word for a real (non-fake) image
LABEL_MAP = {"Realism": "Real", "Deepfake": "Deepfake"}

_model     = None
_processor = None


def load_model():
    """
    Load model + processor into memory.
    Safe to call multiple times — only loads once (singleton).
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print(f"[model] Downloading / loading  {MODEL_ID}")
    print("[model] First run: ~330 MB download, takes ~1 min on a normal connection.")

    _processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    _model     = ViTForImageClassification.from_pretrained(MODEL_ID)
    _model.to(DEVICE).eval()

    print(f"[model] Ready on {DEVICE}")
    return _model, _processor


def _infer(pil_images: list) -> tuple:
    """
    Run inference on a list of PIL Images.
    Averages softmax probabilities across all images (useful for video frames).

    Returns:
        (label: str, confidence: float)   e.g. ("Deepfake", 94.2)
    """
    model, processor = load_model()

    # processor resizes to 224×224 and normalises — no manual transforms needed
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits       # (N, 2)
        probs  = torch.softmax(logits, dim=1) # (N, 2)
        avg    = probs.mean(dim=0)            # (2,)  average across frames

    class_idx  = avg.argmax().item()
    confidence = round(avg[class_idx].item() * 100, 2)
    raw_label  = model.config.id2label[class_idx]   # "Realism" or "Deepfake"
    label      = LABEL_MAP.get(raw_label, raw_label)

    return label, confidence


def predict_image(pil_img: Image.Image) -> tuple:
    """Single image → ("Real" | "Deepfake", confidence %)"""
    return _infer([pil_img])


def predict_video(pil_frames: list) -> tuple:
    """List of PIL frames → ("Real" | "Deepfake", confidence %)"""
    return _infer(pil_frames)