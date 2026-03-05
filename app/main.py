"""
FarmGuard AI — FastAPI Backend
Clean, production-ready rebuild with HuggingFace Transformers pipeline
"""
from dotenv import load_dotenv
load_dotenv()
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.classifier import get_classifier, CROP_FILTERS
from app.data.remedies import get_remedy, SEVERITY_COLORS
from app.services.weather import get_weather, compute_disease_risk
from app.services.history import (
    init_db, record_scan, get_scan_history,
    get_disease_trend, get_summary_stats, delete_scan
)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("🌱 Loading FarmGuard AI model...")
    get_classifier()
    print("✅ Model ready")
    yield
    print("👋 Shutting down FarmGuard AI")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FarmGuard AI",
    description="Intelligent crop disease detection and advisory API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "FarmGuard AI", "version": "2.0.0"}


@app.get("/crops")
def list_crops():
    return {"crops": sorted(CROP_FILTERS.keys())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    crop_age: Optional[int] = Form(0),
    location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    save_history: bool = Form(True)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG, WebP)")

    MAX_SIZE = 10 * 1024 * 1024
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")
    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Image appears to be empty or corrupted.")

    try:
        classifier = get_classifier()
        prediction = classifier.predict(
            image_bytes=image_bytes,
            crop_type=crop_type,
            top_k=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    remedy = get_remedy(prediction["class_name"])
    severity = remedy.get("severity", "moderate")
    severity_color = SEVERITY_COLORS.get(severity, "#f59e0b")

    age = crop_age or 0
    if age < 20:
        growth_stage = "Seedling"
    elif age < 60:
        growth_stage = "Vegetative"
    elif age < 100:
        growth_stage = "Reproductive"
    else:
        growth_stage = "Maturation"

    weather_data = None
    weather_risk = {"level": "unknown", "score": None, "message": "No location provided.", "factors": []}

    if location:
        weather_data = await get_weather(location)
        if weather_data:
            weather_risk = compute_disease_risk(weather_data, remedy.get("weather_risk", {}))

    scan_id = None
    if save_history:
        scan_id = record_scan(
            class_name=prediction["class_name"],
            common_name=remedy["common_name"],
            confidence=prediction["confidence"],
            is_healthy=prediction["is_healthy"],
            severity=severity,
            crop_type=crop_type,
            crop_age=age,
            location=location,
            weather=weather_data,
            notes=notes
        )

    return {
        "scan_id": scan_id,
        "class_name": prediction["class_name"],
        "common_name": remedy["common_name"],
        "confidence": prediction["confidence"],
        "is_healthy": prediction["is_healthy"],
        "alternatives": prediction["alternatives"],
        "crop_filter_applied": prediction["crop_filter_applied"],
        "pathogen": remedy.get("pathogen"),
        "severity": severity,
        "severity_color": severity_color,
        "description": remedy.get("description", ""),
        "immediate_actions": remedy.get("immediate_actions", []),
        "prevention": remedy.get("prevention", []),
        "organic_options": remedy.get("organic_options", []),
        "risk_factors": remedy.get("risk_factors", []),
        "growth_stage": f"{growth_stage} Stage (Day {age})" if age > 0 else growth_stage,
        "weather": weather_data,
        "weather_risk": weather_risk,
    }


# ── History Routes ────────────────────────────────────────────────────────────

@app.get("/history")
def history(
    limit: int = 50,
    crop_type: Optional[str] = None,
    only_diseases: bool = False
):
    return {"scans": get_scan_history(limit=limit, crop_type=crop_type, only_diseases=only_diseases)}


@app.get("/history/trends")
def trends(days: int = 30, crop_type: Optional[str] = None):
    return get_disease_trend(days=days, crop_type=crop_type)


@app.get("/history/stats")
def stats():
    return get_summary_stats()


@app.delete("/history/{scan_id}")
def delete_scan_record(scan_id: int):
    success = delete_scan(scan_id)
    if not success:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"deleted": True, "scan_id": scan_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)