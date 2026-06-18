"""Tests de l'API de segmentation.

Execution :
    pytest api/test_api.py -v
"""

import io
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, CATEGORY_NAMES, IMG_SIZE


@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI partage (context manager -> declenche le lifespan)."""
    with TestClient(app) as c:
        yield c


def make_dummy_image_bytes(size=(64, 64), color=(128, 128, 128)):
    """Genere une image PNG factice en bytes pour les tests."""
    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "endpoints" in body
    assert "/predict" in body["endpoints"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_categories(client):
    response = client.get("/categories")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == 8
    assert len(body["categories"]) == 8
    names = [c["name"] for c in body["categories"]]
    assert names == CATEGORY_NAMES


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict",
        files={"file": ("dummy.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_returns_mask_when_model_loaded(client):
    """Test bout en bout : appelle /predict avec une image factice.

    Skip si le modele n'a pas pu etre charge (CI sans .keras).
    """
    health = client.get("/health").json()
    if not health.get("model_loaded"):
        pytest.skip("Modele non charge dans cet environnement.")

    image_bytes = make_dummy_image_bytes(size=IMG_SIZE)
    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "mask_png_base64" in body
    assert body["mask_shape"] == list(IMG_SIZE)

    decoded = base64.b64decode(body["mask_png_base64"])
    mask_img = Image.open(io.BytesIO(decoded))
    arr = np.array(mask_img)
    assert arr.shape == (IMG_SIZE[0], IMG_SIZE[1], 3)
