"""
LeafGuard AI – Django Views
"""

import json
import base64
import io
import logging
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from .ml_inference import predict, CLASSES

logger = logging.getLogger(__name__)


def index(request):
    """Main page — upload + webcam interface."""
    return render(request, "detector/index.html", {
        "classes": CLASSES,
    })


def about(request):
    """About / documentation page."""
    return render(request, "detector/about.html")


@csrf_exempt
@require_http_methods(["POST"])
def predict_upload(request):
    """
    Endpoint: POST /predict/upload/
    Accepts a multipart image file, returns JSON prediction.
    """
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image file provided."}, status=400)

    img_file = request.FILES["image"]

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if img_file.content_type not in allowed_types:
        return JsonResponse(
            {"error": f"Unsupported file type: {img_file.content_type}. Use JPEG, PNG, or WebP."},
            status=400,
        )

    # Validate size (5 MB max)
    if img_file.size > 5 * 1024 * 1024:
        return JsonResponse({"error": "File too large. Maximum size is 5 MB."}, status=400)

    try:
        image_bytes = img_file.read()
        result = predict(str(settings.MODEL_PATH), image_bytes)
        return JsonResponse({"success": True, "result": result})
    except Exception as e:
        logger.exception("Prediction error (upload)")
        return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_webcam(request):
    """
    Endpoint: POST /predict/webcam/
    Accepts a base64-encoded image (data URL) from webcam, returns JSON prediction.
    """
    try:
        body = json.loads(request.body)
        data_url = body.get("image", "")
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    if not data_url:
        return JsonResponse({"error": "No image data provided."}, status=400)

    try:
        # Strip the data URL prefix: "data:image/jpeg;base64,<data>"
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(data_url)
        result = predict(str(settings.MODEL_PATH), image_bytes)
        return JsonResponse({"success": True, "result": result})
    except Exception as e:
        logger.exception("Prediction error (webcam)")
        return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)


def health_check(request):
    """Simple health-check endpoint."""
    model_path = Path(settings.MODEL_PATH)
    return JsonResponse({
        "status": "ok",
        "model_loaded": model_path.exists(),
        "model_path": str(model_path),
    })
