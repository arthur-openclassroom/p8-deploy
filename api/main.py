"""API de prédiction de segmentation sémantique (Cityscapes, 8 catégories).

Usage :
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Variables d'environnement :
    MODEL_PATH : chemin vers le modèle .keras (défaut : models/best_model.keras)
"""

import io
import os
import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("segmentation_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage (best-effort)."""
    try:
        load_model()
    except FileNotFoundError as exc:
        logger.warning("Démarrage sans modèle : %s", exc)
    yield


app = FastAPI(
    title="Segmentation API - Future Vision Transport",
    description="API de segmentation sémantique d'images pour véhicules autonomes.",
    version="1.0.0",
    lifespan=lifespan,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.keras")
IMG_SIZE = (256, 256)
NUM_CLASSES = 8

CATEGORY_NAMES = [
    "void", "flat", "construction", "object",
    "nature", "sky", "human", "vehicle",
]

CATEGORY_COLORS = [
    [0, 0, 0], [128, 64, 128], [70, 70, 70], [153, 153, 153],
    [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142],
]

model = None


def load_model():
    """Charge le modèle de segmentation (lazy, idempotent)."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}. "
                "Définir la variable d'environnement MODEL_PATH ou "
                "exécuter le notebook pour produire le fichier."
            )
        logger.info("Chargement du modèle depuis %s", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Modèle chargé.")
    return model


def preprocess_image(image_bytes):
    """Prétraite une image pour le modèle.

    Parameters
    ----------
    image_bytes : bytes
        Image en bytes.

    Returns
    -------
    np.ndarray
        Image prétraitée (1, H, W, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def mask_to_rgb(mask):
    """Convertit un masque de catégories en image RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id, color in enumerate(CATEGORY_COLORS):
        rgb[mask == cat_id] = color
    return rgb


def encode_mask_png(mask_rgb):
    """Encode un masque RGB en base64 PNG."""
    img = Image.fromarray(mask_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
async def root():
    """Route racine."""
    return {
        "message": "API de segmentation sémantique - Future Vision Transport",
        "endpoints": {
            "/predict": "POST - Envoyer une image pour obtenir le masque de segmentation",
            "/health": "GET - Vérification de l'état de l'API",
            "/categories": "GET - Liste des catégories",
        },
    }


@app.get("/health")
async def health():
    """Vérification de santé de l'API."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/categories")
async def categories():
    """Retourne la liste des catégories."""
    return {
        "num_classes": NUM_CLASSES,
        "categories": [
            {"id": i, "name": name, "color": color}
            for i, (name, color) in enumerate(zip(CATEGORY_NAMES, CATEGORY_COLORS))
        ],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prédit le masque de segmentation pour une image.

    Parameters
    ----------
    file : UploadFile
        Image au format PNG ou JPEG.

    Returns
    -------
    JSONResponse
        Masque prédit encodé en base64 (PNG) + classes par pixel.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        image_bytes = await file.read()
        orig_w, orig_h = Image.open(io.BytesIO(image_bytes)).size
        input_tensor = preprocess_image(image_bytes)

        mdl = load_model()
        prediction = mdl.predict(input_tensor, verbose=0)

        mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
        mask_rgb = mask_to_rgb(mask)
        # Remet le masque a la taille de l'image d'entree (NEAREST garde les 8 couleurs).
        mask_rgb = np.array(
            Image.fromarray(mask_rgb).resize((orig_w, orig_h), Image.NEAREST)
        )
        mask_b64 = encode_mask_png(mask_rgb)

        return JSONResponse(content={
            "mask_png_base64": mask_b64,
            "mask_shape": [orig_h, orig_w],
            "categories_present": [
                CATEGORY_NAMES[c] for c in np.unique(mask)
            ],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
