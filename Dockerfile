# For data prep + eval + serving. Actual training needs a CUDA base image
# (e.g. nvidia/cuda) + requirements-train.txt on a GPU host.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pytest", "tests/", "-q"]
