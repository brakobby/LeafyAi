"""
LeafGuard AI - ML Inference Module
Handles model loading (lazy, singleton) and image classification.
"""

import numpy as np
import io
import logging
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Class config ────────────────────────────────────────────────────────────
CLASSES = {
    0: {
        "name": "Healthy",
        "color": "#22c55e",
        "bg": "#dcfce7",
        "icon": "✓",
        "description": "No disease detected. The leaf appears healthy.",
        "action": "Continue regular care and monitoring.",
    },
    1: {
        "name": "Early Blight",
        "color": "#f97316",
        "bg": "#fff7ed",
        "icon": "⚠",
        "description": "Alternaria solani fungal infection causing dark concentric spots.",
        "action": "Remove affected leaves, apply copper-based fungicide, improve air circulation.",
    },
    2: {
        "name": "Leaf Mold",
        "color": "#a855f7",
        "bg": "#faf5ff",
        "icon": "⚠",
        "description": "Passalora fulva fungal infection — yellow spots on upper surface, mold beneath.",
        "action": "Reduce humidity, prune affected foliage, apply appropriate fungicide.",
    },
    3: {
        "name": "Tomato Yellow Leaf Curl Virus",
        "color": "#ef4444",
        "bg": "#fef2f2",
        "icon": "✕",
        "description": "Viral disease spread by whiteflies causing leaf curling and yellowing.",
        "action": "Remove infected plants, control whitefly population, use resistant varieties.",
    },
}

IMG_SIZE = (224, 224)

# ── Singleton model holder ───────────────────────────────────────────────────
_model = None
_model_loaded = False


def load_model(model_path: str):
    """Load the Keras model once and cache it."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    # If the model file does not exist, avoid importing TensorFlow (prevents binary crashes)
    model_file = Path(model_path)
    if not model_file.exists():
        logger.info("Model file %s not found — using mock predictions.", model_path)
        _model = None
        _model_loaded = True
        return _model

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(str(model_path))
        _model_loaded = True
        logger.info("LeafGuard model loaded from %s", model_path)
    except Exception as e:
        logger.warning("Could not load model: %s — using mock predictions.", e)
        _model = None
        _model_loaded = True
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize and normalize an image for inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def predict(model_path: str, image_bytes: bytes) -> dict:
    """
    Run inference and return a result dict with class info + probabilities.
    Falls back to mock data if no model is available.
    """
    model = load_model(model_path)

    if model is not None:
        tensor = preprocess_image(image_bytes)
        probs = model.predict(tensor, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100
    else:
        # Mock for development (no model yet)
        import random
        probs = np.random.dirichlet(np.ones(4)).astype(float)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100

    cls = CLASSES[class_idx]
    all_probs = [
        {
            "name": CLASSES[i]["name"],
            "probability": round(float(probs[i]) * 100, 1),
            "color": CLASSES[i]["color"],
        }
        for i in range(4)
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "class_index": class_idx,
        "class_name": cls["name"],
        "confidence": round(confidence, 1),
        "color": cls["color"],
        "bg": cls["bg"],
        "icon": cls["icon"],
        "description": cls["description"],
        "action": cls["action"],
        "is_healthy": class_idx == 0,
        "all_probabilities": all_probs,
        "mock": model is None,
    }
