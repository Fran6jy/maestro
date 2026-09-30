# MAESTRO live trial (OANDA practice only). CPU-only: models are trained elsewhere
# (laptop or Kaggle GPU) and copied into ./state/models/current.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU build of PyTorch (has arm64 wheels, so it runs on an Oracle Ampere box).
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy pandas pyarrow scikit-learn scipy hmmlearn statsmodels \
        requests python-dotenv pyyaml

# The code is the public repository at a pinned commit, so the running version is known.
ARG COMMIT=main
RUN git clone -q https://github.com/Fran6jy/maestro.git /opt/src/maestro \
    && git -C /opt/src/maestro checkout -q "${COMMIT}"

ENV PYTHONPATH=/opt/src \
    MAESTRO_RAW_DIR=/state/raw \
    MAESTRO_DATA_DIR=/state/data \
    MAESTRO_OUTPUT_DIR=/state/out \
    OANDA_ENVIRONMENT=practice \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

WORKDIR /state
ENTRYPOINT ["python", "-m", "maestro.live.loop", "--models", "/state/models/current", "--state", "/state"]
