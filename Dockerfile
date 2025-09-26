FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-distutils \
        build-essential \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV PADDLEOCR_HOME=/root/.paddleocr/whl \
    PADDLEOCR_DET_MODEL_DIR=/root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer \
    PADDLEOCR_REC_MODEL_DIR=/root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer \
    PADDLEOCR_CLS_MODEL_DIR=/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer \
    PADDLEOCR_USE_GPU=1

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
