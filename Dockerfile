FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PADDLEOCR_HOME=/root/.paddleocr

RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False)"

COPY . /app

CMD ["python", "-m", "serverless_handler"]
