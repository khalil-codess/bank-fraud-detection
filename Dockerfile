# One image for both services: the scoring API and the Streamlit dashboard.
# The model files are not in git: run `python -m fraud.train --config configs/sparkov.yaml` first.
FROM python:3.13-slim

# libgomp: OpenMP runtime used by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Pinned versions first (cached layer): pickled models need the versions they were trained with.
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY app.py ./
COPY configs ./configs
COPY artifacts ./artifacts
COPY outputs ./outputs

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
