FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /input /outputs /app/outputs

CMD ["python", "-m", "src.train"]