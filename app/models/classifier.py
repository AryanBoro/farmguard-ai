"""
FarmGuard AI — PyTorch EfficientNet-B4 Classifier
Transfer learning on PlantVillage dataset (38 classes)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from typing import Tuple, List

NUM_CLASSES = 15

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

CROP_FILTERS = {
    "Pepper": [i for i, c in enumerate(CLASS_NAMES) if c.startswith('Pepper')],
    "Potato": [i for i, c in enumerate(CLASS_NAMES) if c.startswith('Potato')],
    "Tomato": [i for i, c in enumerate(CLASS_NAMES) if c.startswith('Tomato')],
}

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B4 with custom classification head.
    
    Architecture:
      - EfficientNet-B4 backbone (ImageNet pretrained)
      - Dropout (0.4) for regularization
      - Linear head → num_classes
    
    EfficientNet-B4 chosen over B0/ResNet because:
      - Better accuracy/params tradeoff at this task scale
      - Input 380×380 captures fine leaf texture detail
      - ~19M params vs ResNet50's 25M with better accuracy
    """
    weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_inference_transforms() -> transforms.Compose:
    """
    Inference preprocessing pipeline.
    Matches ImageNet normalization used during EfficientNet-B4 pretraining.
    """
    return transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_training_transforms() -> transforms.Compose:
    """
    Training augmentation pipeline for PlantVillage fine-tuning.
    Aggressive augmentation since PlantVillage images are lab-controlled
    but real-world images will vary significantly.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(380, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class FarmGuardClassifier:
    """
    Inference wrapper around the EfficientNet-B4 model.
    Handles image loading, preprocessing, and filtered prediction.
    """

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=(model_path is None))
        self.transform = get_inference_transforms()

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Support both raw state_dict and wrapped checkpoints
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Load raw image bytes → normalized tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)  # [1, 3, 380, 380]
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image_bytes: bytes,
        crop_type: str = None,
        top_k: int = 3
    ) -> dict:
        """
        Run inference and return structured prediction result.

        Args:
            image_bytes: Raw uploaded image bytes
            crop_type: Optional crop filter (e.g. "Tomato") to restrict outputs
            top_k: Number of alternative predictions to return

        Returns:
            dict with class_name, confidence, top_k predictions, is_healthy
        """
        tensor = self.preprocess(image_bytes)
        logits = self.model(tensor)[0]  # [num_classes]

        # Apply softmax to get calibrated probabilities
        probs = torch.softmax(logits, dim=0)

        if crop_type and crop_type in CROP_FILTERS:
            # Zero out all non-relevant classes, renormalize within crop group
            allowed_indices = CROP_FILTERS[crop_type]
            mask = torch.zeros_like(probs)
            mask[allowed_indices] = probs[allowed_indices]
            probs = mask / (mask.sum() + 1e-8)

        # Top prediction
        top_conf, top_idx = probs.max(dim=0)
        predicted_class = CLASS_NAMES[top_idx.item()]
        confidence = top_conf.item()

        # Top-K alternatives
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(CLASS_NAMES)))
        alternatives = [
            {
                "class_name": CLASS_NAMES[i.item()],
                "confidence": round(p.item() * 100, 2)
            }
            for p, i in zip(top_k_probs, top_k_indices)
            if i.item() != top_idx.item()
        ]

        return {
            "class_name": predicted_class,
            "confidence": round(confidence * 100, 2),  # Real confidence — no boosting
            "is_healthy": "healthy" in predicted_class.lower(),
            "alternatives": alternatives,
            "crop_filter_applied": crop_type if crop_type in (CROP_FILTERS or {}) else None
        }


# Singleton — loaded once at app startup
_classifier_instance: FarmGuardClassifier = None

def get_classifier(model_path: str = None) -> FarmGuardClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FarmGuardClassifier(model_path=model_path)
    return _classifier_instance
