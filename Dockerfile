# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++


# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY . .

# Expose port and run API
EXPOSE 8000
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]
