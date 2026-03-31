FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    ; apt-get install -y --no-install-recommends libsndfile1 \
    ; rm -rf /var/lib/apt/lists/*

COPY backend/requirements-backend.txt /app/backend/requirements-backend.txt
RUN pip install --no-cache-dir -r /app/backend/requirements-backend.txt

COPY backend /app/backend
COPY model /app/model

EXPOSE 40150

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "40150"]
