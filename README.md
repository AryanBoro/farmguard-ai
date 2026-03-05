<<<<<<< Updated upstream
---
title: FarmGuard AI
emoji: 🌾
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI 
license: mit
tags: ["fastapi", "docker", "crop disease", "ai", "api"]
---
=======
# 🌱 FarmGuard AI v2.0

> Intelligent crop disease detection and advisory system — rebuilt from scratch with PyTorch + FastAPI.

---

## What's New in v2.0

| Feature | v1 (old) | v2 (this) |
|---|---|---|
| ML Framework | TensorFlow/Keras `.hdf5` | **PyTorch EfficientNet-B4** |
| Backend | Flask | **FastAPI** |
| Confidence scores | Faked (`97.5% + boost`) | **Real softmax probabilities** |
| Remedy database | 3 entries | **38 full entries** (all classes) |
| Weather | Hardcoded Haryana | **Any location via API** |
| Crop history | None | **SQLite with trend analytics** |
| API | Basic | **Full REST with validation** |

---

## Project Structure

```
farmguard-ai/
├── app/
│   ├── main.py                 # FastAPI app + all routes
│   ├── models/
│   │   └── classifier.py       # EfficientNet-B4 model + inference wrapper
│   ├── services/
│   │   ├── weather.py          # WeatherAPI integration + risk assessment
│   │   └── history.py          # SQLite crop history + analytics
│   └── data/
│       └── remedies.py         # Full remedy DB (38 disease classes)
├── training/
│   └── train.py                # Training script for PlantVillage dataset
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set environment variables

```bash
export WEATHER_API_KEY="your_key_from_weatherapi.com"
export MODEL_PATH="/path/to/farmguard_best.pt"   # Leave unset to use ImageNet pretrained weights
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Training the Model

### Get the dataset

Download PlantVillage from Kaggle:
```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

Expected structure:
```
PlantVillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    ... (38 class folders)
```

### Run training

```bash
# Basic training (30 epochs, GPU recommended)
python training/train.py --data_dir /path/to/PlantVillage --epochs 30

# With mixed precision (faster on RTX/A-series GPUs)
python training/train.py --data_dir /path/to/PlantVillage --epochs 30 --amp

# Full options
python training/train.py \
  --data_dir /path/to/PlantVillage \
  --output_dir ./checkpoints \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --freeze_epochs 5 \
  --amp
```

### Training Strategy

Two-phase training:
1. **Epochs 1–5 (head only):** Backbone frozen. Only the classification head trains. Fast convergence, prevents destroying pretrained features.
2. **Epochs 6–30 (full fine-tune):** Full model unfreezes with differential learning rates — backbone at `lr × 0.1`, head at `lr`. Allows backbone to adapt to leaf texture features.

**Expected results on PlantVillage:**
- Validation accuracy: ~97–99%
- Training time: ~2–4 hours on a single GPU (RTX 3080+)

### After training

```bash
export MODEL_PATH="./checkpoints/farmguard_best.pt"
uvicorn app.main:app --reload
```

---

## API Reference

### `POST /predict`

Main prediction endpoint.

**Form fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image | ✅ | Leaf image (JPEG/PNG/WebP, max 10MB) |
| `crop_type` | string | ❌ | e.g. `"Tomato"` — filters predictions to this crop |
| `crop_age` | int | ❌ | Days since planting (used for growth stage) |
| `location` | string | ❌ | City/region for weather risk assessment |
| `notes` | string | ❌ | Field notes saved with history entry |
| `save_history` | bool | ❌ | Default `true` — saves scan to history |

**Response:**
```json
{
  "scan_id": 42,
  "class_name": "Tomato___Early_blight",
  "common_name": "Tomato Early Blight",
  "confidence": 94.7,
  "is_healthy": false,
  "alternatives": [
    { "class_name": "Tomato___Target_Spot", "confidence": 3.2 },
    { "class_name": "Tomato___Septoria_leaf_spot", "confidence": 1.1 }
  ],
  "pathogen": "Alternaria solani (fungus)",
  "severity": "moderate",
  "severity_color": "#f59e0b",
  "description": "...",
  "immediate_actions": ["..."],
  "prevention": ["..."],
  "organic_options": ["..."],
  "growth_stage": "Vegetative Stage (Day 45)",
  "weather": {
    "location": "Ludhiana, India",
    "temp_c": 27,
    "humidity": 78,
    "precip_mm": 0,
    "condition": "Partly cloudy"
  },
  "weather_risk": {
    "level": "moderate",
    "score": 55,
    "message": "🟡 Weather conditions present moderate disease risk.",
    "factors": ["High humidity (78%) favors pathogen spread"]
  }
}
```

### `GET /history`

Query params: `limit`, `crop_type`, `only_diseases`

### `GET /history/trends`

Query params: `days` (default 30), `crop_type`

### `GET /history/stats`

Returns dashboard summary: total scans, disease rate, top diseases.

### `GET /crops`

Returns list of supported crop types.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WEATHER_API_KEY` | (empty) | From [weatherapi.com](https://weatherapi.com) — free tier works |
| `MODEL_PATH` | `None` | Path to `farmguard_best.pt`. If unset, uses ImageNet pretrained weights (lower accuracy until fine-tuned) |

---

## Frontend Integration (Lovable)

The API is fully CORS-enabled. Key endpoint for the frontend:

```
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```

All responses are JSON. The `severity_color` field provides hex color codes ready for UI use.
>>>>>>> Stashed changes
