# Stage 1: Builder (빌드 도구 포함)
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (실행에 필요한 것만)
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY config/ ./config/
RUN mkdir -p data/inputs data/outputs
EXPOSE 8080
CMD ["uvicorn", "src.process_api:app", "--host", "0.0.0.0", "--port", "8080"]
