---
title: P8 Segmentation API
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# P8 Segmentation API

API FastAPI de segmentation sémantique d'images (Cityscapes, 8 catégories).

- `POST /predict` — envoie une image, retourne le masque prédit (PNG base64)
- `GET /health` — état du service
- `GET /categories` — liste des catégories
