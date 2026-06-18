FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install -r /app/webapp/requirements.txt

COPY webapp /app/webapp

ENV API_URL=http://localhost:8000
ENV PORT=8501

EXPOSE 8501

CMD ["sh", "-c", "streamlit run webapp/app.py --server.port ${PORT} --server.address 0.0.0.0"]
