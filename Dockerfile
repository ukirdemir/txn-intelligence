FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Serving needs no torch (it is an optional extra), which keeps the image small.
RUN pip install --no-cache-dir .

ENV MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000
CMD ["uvicorn", "txn_intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
