---
title: FarmGuard AI
emoji: 🌾
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI backend for intelligent crop disease detection and advisory API
license: mit
tags: ["fastapi", "docker", "crop disease", "ai", "api"]
---

## 🌱 FarmGuard AI

FarmGuard AI is a FastAPI backend for crop disease detection with real confidence scores, weather risk analysis, and remedy suggestions. This API accepts plant leaf images and returns classification results and advisory data.

### 🚀 Endpoints

- **GET /** – Health check  
- **GET /crops** – List available crop types  
- **POST /predict** – Predict disease from image  
- **GET /history** – Scan history  
- **GET /history/stats** – Summary statistics  
- **DELETE /history/{scan_id}** – Delete a scan record

### 🛠 Deployment

This Space runs inside a Docker container. The FastAPI server listens on port **7860** (configured above), so Hugging Face can route traffic properly. :contentReference[oaicite:0]{index=0}

You can explore the API interactively at `/docs` once the Space is running.

---

### 📚 Documentation

Check out the full configuration reference here:  
https://huggingface.co/docs/hub/spaces-config-reference

---

### 🧠 Credits

Powered by PyTorch and FastAPI.