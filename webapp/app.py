"""Application Streamlit de démonstration de la segmentation sémantique.

Usage :
    streamlit run webapp/app.py

L'application permet de :
- Sélectionner une image du dataset par son ID
- Lancer la prédiction via l'API
- Afficher l'image originale, le masque réel et le masque prédit
"""

import os
import io
import glob
import base64

import requests
import numpy as np
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
GTFINE_DIR = os.environ.get(
    "GTFINE_DIR",
    os.path.join(os.path.dirname(__file__), "sample", "gtFine"),
)
IMAGES_DIR = os.environ.get(
    "IMAGES_DIR",
    os.path.join(os.path.dirname(__file__), "sample", "leftImg8bit"),
)

CATEGORY_NAMES = [
    "void", "flat", "construction", "object",
    "nature", "sky", "human", "vehicle",
]
CATEGORY_COLORS = [
    (0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153),
    (107, 142, 35), (70, 130, 180), (220, 20, 60), (0, 0, 142),
]

LABEL_TO_CATEGORY = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 1, 9: 1, 10: 1,
    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2,
    17: 3, 18: 3, 19: 3, 20: 3,
    21: 4, 22: 4,
    23: 5,
    24: 6, 25: 6,
    26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7, 32: 7, 33: 7,
    -1: 0, 255: 0,
}


def mask_to_rgb(mask):
    """Convertit un masque de catégories en image RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id, color in enumerate(CATEGORY_COLORS):
        rgb[mask == cat_id] = color
    return rgb


def map_labels_to_categories(mask):
    """Convertit les labels fins en 8 catégories."""
    mapped = np.zeros_like(mask, dtype=np.uint8)
    for label_id, cat_id in LABEL_TO_CATEGORY.items():
        if label_id >= 0:
            mapped[mask == label_id] = cat_id
    mapped[mask == 255] = 0
    return mapped


def get_available_images(split="val"):
    """Récupère la liste des images disponibles."""
    mask_pattern = os.path.join(GTFINE_DIR, split, "*", "*_gtFine_labelIds.png")
    mask_paths = sorted(glob.glob(mask_pattern))
    image_ids = []
    for mask_path in mask_paths:
        basename = os.path.basename(mask_path)
        img_id = basename.replace("_gtFine_labelIds.png", "")
        image_ids.append(img_id)
    return image_ids, mask_paths


def get_image_paths(image_id, split="val"):
    """Retourne les chemins de l'image et du masque pour un ID donné."""
    city = image_id.split("_")[0]
    img_path = os.path.join(IMAGES_DIR, split, city, f"{image_id}_leftImg8bit.png")
    mask_path = os.path.join(GTFINE_DIR, split, city, f"{image_id}_gtFine_labelIds.png")
    return img_path, mask_path


def call_api(image_bytes):
    """Appelle l'API de prédiction."""
    files = {"file": ("image.png", image_bytes, "image/png")}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Segmentation Cityscapes - Future Vision Transport",
    layout="wide",
)

st.title("Segmentation sémantique - Future Vision Transport")
st.markdown(
    "Application de démonstration du modèle de segmentation d'images "
    "pour véhicules autonomes (8 catégories)."
)

st.sidebar.header("Configuration")
split = st.sidebar.selectbox("Split du dataset", ["val", "train"])

image_ids, mask_paths = get_available_images(split)

if not image_ids:
    st.error(f"Aucune image trouvee dans le split '{split}'.")
    st.stop()

selected_id = st.sidebar.selectbox(
    f"Image ({len(image_ids)} disponibles)",
    image_ids,
    format_func=lambda x: x,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Légende des catégories")
for i, (name, color) in enumerate(zip(CATEGORY_NAMES, CATEGORY_COLORS)):
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    st.sidebar.markdown(
        f'<span style="color:{hex_color}; font-size:20px;">&#9632;</span> {name}',
        unsafe_allow_html=True,
    )

img_path, mask_path = get_image_paths(selected_id, split)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Image originale")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Image non trouvee : {img_path}")

with col2:
    st.subheader("Masque réel")
    if os.path.exists(mask_path):
        mask_raw = np.array(Image.open(mask_path))
        mask_cat = map_labels_to_categories(mask_raw)
        mask_rgb = mask_to_rgb(mask_cat)
        st.image(mask_rgb, use_container_width=True)
    else:
        st.warning("Masque non disponible.")

with col3:
    st.subheader("Masque prédit")
    if st.button("Lancer la prédiction"):
        if os.path.exists(img_path):
            with st.spinner("Prédiction en cours..."):
                try:
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()
                    result = call_api(image_bytes)
                    mask_b64 = result["mask_png_base64"]
                    mask_pred = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    st.image(mask_pred, use_container_width=True)
                    st.success(
                        f"Catégories détectées : {', '.join(result['categories_present'])}"
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Impossible de se connecter a l'API ({API_URL}). "
                        "Assurez-vous que l'API est lancée avec : "
                        "uvicorn api.main:app --port 8000"
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.warning("L'image source n'est pas disponible pour la prédiction.")

st.markdown("---")
st.subheader("Tester avec une image personnalisée")
uploaded_file = st.file_uploader(
    "Choisir une image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.image(uploaded_file, caption="Image importée", use_container_width=True)
    with col_up2:
        if st.button("Prédire le masque"):
            with st.spinner("Prédiction en cours..."):
                try:
                    result = call_api(uploaded_file.getvalue())
                    mask_b64 = result["mask_png_base64"]
                    mask_pred = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
                    st.image(
                        mask_pred,
                        caption="Masque prédit",
                        use_container_width=True,
                    )
                    st.success(
                        f"Catégories détectées : {', '.join(result['categories_present'])}"
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Impossible de se connecter a l'API ({API_URL}). "
                        "Vérifiez que l'API est lancée."
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
