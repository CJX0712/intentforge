# ---- runtime -------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INTENTFORGE_RANDOM_SEED=42

WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# Run the end-to-end demo (train + evaluate + benchmark -> benchmark.json)
CMD ["python", "intentforge/examples/run_demo.py"]
