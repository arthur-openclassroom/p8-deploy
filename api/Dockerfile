FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

COPY api /app/api
# On n'embarque que le modele leger de deploiement (~52 Mo), pas tout models/.
COPY models/model_deploy.keras /app/models/model_deploy.keras

ENV MODEL_PATH=/app/models/model_deploy.keras
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
