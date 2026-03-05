"""
FarmGuard AI — Inference Pipeline
Uses linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification from HuggingFace
38 PlantVillage classes, no local model file needed.
"""

import io
from typing import Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# ── Model ID ──────────────────────────────────────────────────────────────────
HF_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# ── Crop filters ──────────────────────────────────────────────────────────────
# Maps crop type to valid class index ranges for filtering predictions
CROP_FILTERS = {
    "Apple":      list(range(0, 4)),
    "Blueberry":  [4],
    "Cherry":     list(range(5, 7)),
    "Corn":       list(range(7, 11)),
    "Grape":      list(range(11, 15)),
    "Orange":     [15],
    "Peach":      list(range(16, 18)),
    "Pepper":     list(range(18, 20)),
    "Potato":     list(range(20, 23)),
    "Raspberry":  [23],
    "Soybean":    [24],
    "Squash":     [25],
    "Strawberry": list(range(26, 28)),
    "Tomato":     list(range(28, 38)),
}

# ── Singleton ─────────────────────────────────────────────────────────────────
_classifier = None


class FarmGuardClassifier:
    def __init__(self):
        print(f"🌱 Loading model from HuggingFace: {HF_MODEL_ID}")
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.id2label = self.model.config.id2label
        print(f"✅ Model ready on {self.device} | Classes: {len(self.id2label)}")

    def predict(
        self,
        image_bytes: bytes,
        crop_type: Optional[str] = None,
        top_k: int = 3
    ) -> dict:
        # ── Load image ────────────────────────────────────────────────────────
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ── Preprocess ────────────────────────────────────────────────────────
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (1, num_classes)

        # ── Crop filter ──────────────────────────────────────────────────────
        crop_filter_applied = False
        if crop_type and crop_type in CROP_FILTERS:
            valid_indices = CROP_FILTERS[crop_type]
            mask = torch.full(logits.shape, float('-inf'), device=self.device)
            for idx in valid_indices:
                if idx < logits.shape[1]:
                    mask[0][idx] = logits[0][idx]
            logits = mask
            crop_filter_applied = True

        # ── Softmax + top-k ──────────────────────────────────────────────────
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        # ── Build result ─────────────────────────────────────────────────────
        best_idx = top_indices[0]
        best_label = self.id2label[best_idx]
        best_conf = round(top_probs[0] * 100, 2)
        is_healthy = "healthy" in best_label.lower()

        alternatives = [
            {
                "class_name": self.id2label[idx],
                "confidence": round(prob * 100, 2)
            }
            for idx, prob in zip(top_indices[1:], top_probs[1:])
        ]

        return {
            "class_name": best_label,
            "confidence": best_conf,
            "is_healthy": is_healthy,
            "alternatives": alternatives,
            "crop_filter_applied": crop_filter_applied,
        }


def get_classifier(model_path: Optional[str] = None) -> FarmGuardClassifier:
    """Return singleton classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = FarmGuardClassifier()
    return _classifier