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

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages \
        libcudnn8 \
        libcudnn8-dev \
        libnccl2 \
        libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/cu121 \
    && python -c "import paddle; paddle.utils.run_check()"

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
