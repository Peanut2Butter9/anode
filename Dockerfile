# AnoDe web platform for Hugging Face Spaces (Docker SDK, CPU).
# The model bundle is downloaded at startup from the Hub repo in ANODE_BUNDLE_REPO.

FROM node:22-slim AS frontend
WORKDIR /src/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch HF_HOME=/tmp/hf \
    ANODE_REPOS=/opt/repos ANODE_FRONTEND=/app/web/frontend/dist ANODE_DATA=/tmp/anode
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first, so the anode package does not pull the CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY web/server/requirements.txt web/server/requirements.txt
RUN pip install -r web/server/requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Official CSAD code at the commit used in the benchmark
RUN git clone -q https://github.com/Tokichan/CSAD /opt/repos/CSAD \
    && git -C /opt/repos/CSAD checkout -q d46cf856b99bf056088b5f510fd984e04813583a
# ImageNet WideResNet-50 weights (PatchCore backbone), baked into the image
RUN python -c "import torchvision; torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')" \
    && chmod -R a+rX /opt/torch /opt/repos && chmod -R a+w /opt/repos

COPY web/server ./web/server
COPY --from=frontend /src/web/frontend/dist ./web/frontend/dist

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
EXPOSE 7860
CMD ["uvicorn", "anode_web.app:app", "--app-dir", "web/server", "--host", "0.0.0.0", "--port", "7860"]
