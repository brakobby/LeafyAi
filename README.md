# 🌿 LeafGuard AI
**Automated Tomato Plant Disease Detection System**  
*Science Fair Project — Computer Science / Agricultural Technology*  
*Prepared by Nhyira · February 2026*

---

## Overview
LeafGuard AI is a Django web application that uses a custom CNN to classify tomato leaf photos into 4 classes:

| Class | Type | Color |
|-------|------|-------|
| ✅ Healthy | — | Green |
| 🟠 Early Blight | Fungal | Orange |
| 🟣 Leaf Mold | Fungal | Purple |
| 🔴 TYLCV | Viral | Red |

---

## Project Structure
```
leafguard/
├── leafguard/          # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detector/           # Main app
│   ├── views.py        # Image upload + webcam endpoints
│   ├── urls.py
│   ├── ml_inference.py # Model loading & prediction
│   └── templates/
│       └── detector/
│           ├── index.html   # Main UI
│           └── about.html   # About page
├── ml/
│   └── train.py        # CNN training script
├── manage.py
└── requirements.txt
```

---

## Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (Google Colab recommended)

1. Download the PlantVillage dataset from Kaggle:  
   `kaggle datasets download -d emmarex/plantdisease`

2. Unzip and place the 4 tomato folders in `data/PlantVillage/`:
   - `Tomato_healthy/`
   - `Tomato_Early_blight/`
   - `Tomato_Leaf_Mold/`
   - `Tomato_Tomato_Yellow_Leaf_Curl_Virus/`

3. Run training:
   ```bash
   python ml/train.py
   ```
   The best model is saved to `ml/leafguard_model.h5`.

> **No model yet?** The app still runs with mock (random) predictions — useful for testing the UI.

### Step 3 — Run the web app
```bash
python manage.py runserver
```
Open: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI (upload + webcam) |
| `/about/` | GET | Project info |
| `/predict/upload/` | POST | Classify an uploaded image |
| `/predict/webcam/` | POST | Classify a base64 webcam frame |
| `/health/` | GET | Server health check |

### Example: Upload request
```bash
curl -X POST http://127.0.0.1:8000/predict/upload/ \
  -F "image=@leaf.jpg"
```

### Example: Response
```json
{
  "success": true,
  "result": {
    "class_name": "Early Blight",
    "confidence": 94.3,
    "color": "#f97316",
    "description": "Alternaria solani fungal infection...",
    "action": "Remove affected leaves, apply copper-based fungicide...",
    "is_healthy": false,
    "all_probabilities": [...]
  }
}
```

---

## CNN Architecture

```
Input (224×224×3)
  → Block 1: Conv32 × 2 → BN → MaxPool → Dropout(0.25)
  → Block 2: Conv64 × 2 → BN → MaxPool → Dropout(0.25)
  → Block 3: Conv128 × 2 → BN → MaxPool → Dropout(0.30)
  → Block 4: Conv256    → BN → MaxPool → Dropout(0.30)
  → GlobalAvgPool
  → Dense(256) → BN → Dropout(0.5)
  → Softmax(4)
```

- **Loss:** Sparse Categorical Cross-Entropy  
- **Optimizer:** Adam (lr=0.001)  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## Performance Targets (SRS)
- Accuracy ≥ 90% on held-out test set (300 images)
- Inference ≤ 50 ms per frame
- Live FPS ≥ 15
- Model ≤ 200 MB
- Memory ≤ 600 MB during inference

---

## References
- PlantVillage Dataset: https://plantvillage.psu.edu/
- TensorFlow: https://www.tensorflow.org/
- Django: https://www.djangoproject.com/
- IEEE Std 830-1998
