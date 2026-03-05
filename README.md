# 🌱 FarmGuard AI — Backend

> Production-ready FastAPI backend for intelligent crop disease detection, powered by HuggingFace Transformers.

![FarmGuard AI](https://img.shields.io/badge/FarmGuard-AI-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge)

---

## Overview

FarmGuard AI is a crop disease detection API that accepts leaf images and returns AI-powered disease diagnosis, treatment recommendations, weather risk assessment, and scan history analytics.

**Live API:** `https://borreooo-farmguard-ai.hf.space`  
**API Docs:** `https://borreooo-farmguard-ai.hf.space/docs`  
**Frontend:** [farmguard-ai-frontend](https://github.com/AryanBoro/farmguard-ai-frontend)

---

## Features

- 🔍 **Disease Detection** — 38 plant disease classes across 14 crops using MobileNetV2
- 💊 **Remedy Database** — Full treatment advisory for every disease class
- 🌤️ **Weather Risk** — Real-time weather-based disease risk scoring via OpenWeatherMap
- 📊 **Scan History** — SQLite-backed scan history with trend analytics
- ⚡ **Fast Inference** — HuggingFace Transformers pipeline, no local model file needed

---

## Supported Crops & Diseases

| Crop | Diseases |
|---|---|
| Apple | Scab, Black Rot, Cedar Apple Rust |
| Cherry | Powdery Mildew |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight |
| Grape | Black Rot, Black Measles, Leaf Blight |
| Orange | Citrus Greening |
| Peach | Bacterial Spot |
| Pepper | Bacterial Spot |
| Potato | Early Blight, Late Blight |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus |

---

## Tech Stack

- **FastAPI** — REST API framework
- **HuggingFace Transformers** — MobileNetV2 inference pipeline
- **PyTorch** — Model backend
- **SQLite** — Scan history persistence
- **httpx** — Async weather API calls
- **Docker** — Containerized deployment on HuggingFace Spaces

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/crops` | List supported crop types |
| POST | `/predict` | Submit leaf image for diagnosis |
| GET | `/history` | Get scan history |
| GET | `/history/stats` | Dashboard summary stats |
| GET | `/history/trends` | Daily disease trend data |
| DELETE | `/history/{scan_id}` | Delete a scan record |

### POST /predict

```bash
curl -X POST https://borreooo-farmguard-ai.hf.space/predict \
  -F "file=@leaf.jpg" \
  -F "crop_type=Tomato" \
  -F "crop_age=45" \
  -F "location=Punjab"
```

**Response:**
```json
{
  "class_name": "Tomato___Early_blight",
  "common_name": "Tomato Early Blight",
  "confidence": 94.7,
  "is_healthy": false,
  "severity": "moderate",
  "immediate_actions": ["..."],
  "prevention": ["..."],
  "organic_options": ["..."],
  "weather_risk": { "level": "high", "score": 75 },
  "growth_stage": "Vegetative Stage (Day 45)"
}
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/AryanBoro/farmguard-ai
cd farmguard-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
Create a `.env` file:
```
WEATHER_API_KEY=your_openweathermap_key
```

### 4. Run locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`

---

## Deployment

Deployed on **HuggingFace Spaces** using Docker. Every push to `main` triggers an automatic redeploy.

```
Dockerfile → python:3.9-slim
Port: 7860
```

---

## Project Structure

```
farmguard-ai/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py              # FastAPI app + all endpoints
│   ├── models/
│   │   └── classifier.py    # HuggingFace inference pipeline
│   ├── services/
│   │   ├── history.py       # SQLite scan history
│   │   └── weather.py       # OpenWeatherMap integration
│   └── data/
│       └── remedies.py      # Disease remedy database (38 classes)
└── training/
    └── train.py             # EfficientNet-B4 training script (PlantVillage)
```

---

## Related

- 🎨 **Frontend repo:** [farmguard-ai-frontend](https://github.com/AryanBoro/farmguard-ai-frontend)
- 🤗 **Live Space:** [borreooo/farmguard-ai](https://huggingface.co/spaces/borreooo/farmguard-ai)
