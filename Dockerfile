FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/artifacts

ENV PORT=8080
ENV NEXUS_ARTIFACTS_DIR=/app/artifacts

CMD ["sh", "-lc", "python -m nexus_quant dashboard --artifacts ${NEXUS_ARTIFACTS_DIR:-/app/artifacts} --host 0.0.0.0 --port ${PORT:-8080}"]
