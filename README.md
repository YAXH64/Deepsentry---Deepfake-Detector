# 🛡️ DeepSentry — AI Deepfake Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-5A1870?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-E91E8C?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-ViT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-7B2D8B?style=for-the-badge)

**Detect deepfakes in images and videos using a fine-tuned Vision Transformer (ViT) model — ~92% accuracy.**

[Demo](#-demo) · [Features](#-features) · [Quick Start](#-quick-start) · [API Docs](#-api-reference) · [How It Works](#-how-it-works)

</div>

---

## ✨ Features

- 🖼️ **Image detection** — JPG, PNG, WEBP, BMP
- 🎬 **Video detection** — MP4, AVI, MOV, MKV, WEBM (samples up to 20 frames)
- 🤖 **ViT-powered** — `prithivMLmods/Deep-Fake-Detector-v2-Model` (~92% accuracy on 56k test images)
- ⚡ **Fast REST API** — FastAPI + Uvicorn with auto-generated Swagger UI at `/docs`
- 🎨 **Polished frontend** — drag-and-drop SPA, confidence ring, live activity log — no build step required
- 🔁 **Frame averaging** — video predictions average softmax probabilities across all sampled frames for robustness
- 💾 **Cached model weights** — ~330 MB downloaded once, then loaded from `~/.cache/huggingface/` on every run

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YAXH64/Deepsentry---Deepfake-Detector.git
cd deepsentry
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
python main.py
```

> **First run only:** the model weights (~330 MB) are downloaded automatically and cached. This takes ~1 minute depending on your connection. All subsequent starts are near-instant.

### 5. Open the frontend

Open `index.html` directly in your browser — no web server needed.

```
Backend API  →  http://localhost:8000
Swagger UI   →  http://localhost:8000/docs
```

---

## 📁 Project Structure

```
deepsentry/
├── index.html        # Frontend SPA — drag-and-drop UI, results panel
├── main.py           # FastAPI app — routes, CORS, response schema
├── model.py          # ViT model loader + inference engine
├── processor.py      # File decoding, video frame extraction
└── requirements.txt  # Python dependencies
```

---

## 🔌 API Reference

All endpoints return JSON. Full interactive docs at `http://localhost:8000/docs`.

### `GET /`
Health check.
```json
{ "status": "ok" }
```

---

### `POST /detect/image`
Upload an image and get a deepfake prediction.

**Request:** `multipart/form-data`
| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | File | ✅ | `.jpg` `.jpeg` `.png` `.webp` `.bmp` |

**Response:**
```json
{
  "label":      "Real",
  "confidence": 94.72,
  "media_type": "image",
  "elapsed_ms": 312.5
}
```

---

### `POST /detect/video`
Upload a video and get a deepfake prediction. Up to 20 evenly-spaced frames are sampled and averaged.

**Request:** `multipart/form-data`
| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | File | ✅ | `.mp4` `.avi` `.mov` `.mkv` `.webm` |

**Response:**
```json
{
  "label":      "Deepfake",
  "confidence": 87.13,
  "media_type": "video",
  "elapsed_ms": 4821.0
}
```

---

### Error Codes

| Code | Cause |
|------|-------|
| `400` | No file, unsupported format, corrupted file, or no extractable frames |
| `422` | FastAPI validation error (missing required field) |
| `500` | Unhandled server error during inference |

---

## 🧠 How It Works

DeepSentry uses a 4-step analysis pipeline:

```
┌─────────────────────┐
│  1. Facial Geometry  │  Analyzes 3D facial structure and landmark positions
├─────────────────────┤
│  2. Temporal Check   │  Frame-to-frame coherence (videos only)
├─────────────────────┤
│  3. Artifact Scan    │  GAN/diffusion model pixel-level fingerprints
├─────────────────────┤
│  4. ViT Inference    │  Final classification with confidence score
└─────────────────────┘
```

Under the hood, steps 1–3 are surfaced in the UI as animated checks. Step 4 is the actual model inference:

1. Uploaded file → decoded by OpenCV → converted to PIL Image
2. `ViTImageProcessor` resizes to 224×224 and normalizes pixel values
3. ViT forward pass → softmax probabilities over `["Realism", "Deepfake"]`
4. For video: probabilities are averaged across all sampled frames
5. Highest-probability class returned as the label with confidence %

### Model

| Property | Value |
|----------|-------|
| Model ID | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| Architecture | ViT (vit-base-patch16-224-in21k fine-tuned) |
| Accuracy | ~92% on 56,001 test images |
| Input size | 224 × 224 px |
| Labels | `Realism` → **Real** · `Deepfake` → **Deepfake** |
| Device | CUDA (if available) · CPU fallback |

---

## ⚙️ Configuration

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `MAX_FRAMES` | `processor.py` | `20` | Max video frames sampled per upload |
| `MODEL_ID` | `model.py` | `prithivMLmods/...` | HuggingFace model identifier |
| `DEVICE` | `model.py` | auto | `cuda` if available, else `cpu` |
| `host` | `main.py` | `0.0.0.0` | Uvicorn bind address |
| `port` | `main.py` | `8000` | Uvicorn port |
| `API_BASE` | `index.html` | `http://localhost:8000` | Backend URL used by the frontend |

> **Deploying remotely?** Update `API_BASE` in the `<script>` block of `index.html` to point to your server's address.

---

## 📦 Dependencies

```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.7
torch>=2.0.0
transformers>=4.35.0
Pillow>=9.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pydantic>=2.0.0
```

---

## ⚠️ Known Limitations

- **No file size limit** — large video uploads are read fully into RAM. Add a size guard before production use.
- **CORS is open** — `allow_origins=["*"]` is set for local dev. Restrict this before deploying publicly.
- **Single file at a time** — batch upload is not currently supported.
- **Model accuracy** — ~92% means roughly 1 in 12 predictions may be incorrect. Do not use as a sole source of truth.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ · Powered by <a href="https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model">prithivMLmods/Deep-Fake-Detector-v2-Model</a></sub>
</div>
