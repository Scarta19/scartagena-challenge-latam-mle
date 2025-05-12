# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y gcc g++

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all app code + modelo
COPY . .

# Cloud Run requires your container to listen on port 8080
EXPOSE 8080

# Use Gunicorn with Uvicorn worker
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "challenge.api:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
