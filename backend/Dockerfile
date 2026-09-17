FROM python:3.12-slim

# Keep the venv outside /app so the dev bind mount doesn't hide it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Weights come from a GitHub release rather than git. Its own layer, so editing
# code doesn't re-download 63 MB. Override the defaults in the script with
# --build-arg MODEL_URL=... --build-arg MODEL_SHA256=... to build on other weights.
ARG MODEL_URL
ARG MODEL_SHA256
COPY scripts/fetch_model.py ./scripts/
RUN python scripts/fetch_model.py

COPY . .

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
