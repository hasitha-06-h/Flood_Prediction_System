# Flood_Prediction_System/Dockerfile
#
# Multi-stage-ish build isn't really necessary at this project's size, so
# this stays a single stage: install deps, copy source, train the model
# once at build time so the image is immediately servable, then run under
# gunicorn instead of Flask's dev server.

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout, which makes
# container logs show up immediately instead of being buffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages needed to build scikit-learn / xgboost wheels on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so models/floods.save exists inside the
# image itself - avoids a race where the app starts before training
# finishes on first boot.
RUN python train_model.py

EXPOSE 5000

# Render (and most PaaS platforms) inject $PORT at runtime; gunicorn reads
# it via the shell form of CMD so the container works both locally
# (defaults to 5000, see config.py) and on Render.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app
