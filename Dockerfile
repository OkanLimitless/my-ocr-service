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
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV PADDLEOCR_HOME=/root/.paddleocr/whl \
    PADDLEOCR_DET_MODEL_DIR=/root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer \
    PADDLEOCR_REC_MODEL_DIR=/root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer \
    PADDLEOCR_CLS_MODEL_DIR=/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer

RUN mkdir -p /root/.paddleocr/whl/det/en \
    /root/.paddleocr/whl/rec/en \
    /root/.paddleocr/whl/cls \
    && curl -L "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar" -o /tmp/en_det.tar \
    && tar -xf /tmp/en_det.tar -C /root/.paddleocr/whl/det/en \
    && rm /tmp/en_det.tar \
    && curl -L "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar" -o /tmp/en_rec.tar \
    && tar -xf /tmp/en_rec.tar -C /root/.paddleocr/whl/rec/en \
    && rm /tmp/en_rec.tar \
    && curl -L "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" -o /tmp/cls.tar \
    && tar -xf /tmp/cls.tar -C /root/.paddleocr/whl/cls \
    && rm /tmp/cls.tar

COPY . /app

CMD ["python", "-m", "serverless_handler"]
