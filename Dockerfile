FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r /tmp/requirements.txt

COPY . /workspace

CMD ["bash"]
