# FinGuard AI: API + dashboard (Phase 5).
#
# The image holds the code, the prompts, the indicator map and the project documents. The data,
# the Phase 2 model files and the knowledge-base index are NOT in the image (licence and size):
# docker-compose.yml mounts ./data and ./artifacts. Without them, run the synthetic demo world
# (FINGUARD_WORLD=synthetic), which is what CI does.
#
#   docker compose up --build                     # real world (needs data/, artifacts/, the index)
#   FINGUARD_WORLD=synthetic docker compose up    # synthetic demo world
#
# INSTALL_KB=true adds sentence-transformers with a CPU-only PyTorch, needed when the deployed
# index uses a neural embedder or cross-encoder (FINGUARD_EMBEDDER / FINGUARD_RERANKER).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/finguard/.cache/huggingface \
    FINGUARD_HOST=0.0.0.0 \
    FINGUARD_PORT=8000

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 finguard
WORKDIR /app

ARG INSTALL_KB=false
# constraints.txt (optional, made on the machine that trained the model) pins numpy, pandas,
# scikit-learn, scipy, lightgbm and pyarrow to the versions that wrote the pickled artifacts
COPY pyproject.toml README.md constraints.tx[t] ./
COPY src ./src
# editable install: the project root (data/, artifacts/, corpus/) is found from the source tree
RUN C=""; if [ -f constraints.txt ]; then C="-c constraints.txt"; fi \
    && pip install $C -e ".[api]" \
    && if [ "$INSTALL_KB" = "true" ]; then \
         pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install $C -e ".[api,kb]"; \
       fi

COPY corpus ./corpus
COPY cases ./cases
# the Hugging Face cache must exist (owned by finguard) so the named volume mounted there inherits it
RUN mkdir -p data artifacts reports/tables /home/finguard/.cache/huggingface \
    && chown -R finguard:finguard /app /home/finguard

USER finguard
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-m", "finguard.api", "healthcheck"]
CMD ["python", "-m", "finguard.api", "serve"]
