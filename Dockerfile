FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY rf_feature_engineering.py ./
COPY serve_model_api.py ./
COPY random_forest_pipeline.pkl ./

EXPOSE 8000

CMD ["uvicorn", "serve_model_api:app", "--host", "0.0.0.0", "--port", "8000"]
