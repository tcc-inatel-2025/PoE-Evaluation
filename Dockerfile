FROM python:3.7-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install runtime + build deps (compiler, headers, git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc git curl ca-certificates \
      libssl-dev libffi-dev python3-dev nano && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip/tools so it can fetch wheels and build correctly
RUN pip install --upgrade pip setuptools wheel

# Install normal Python dependencies
RUN pip install --no-cache-dir requests radon pylint pandas matplotlib seaborn

# Clone HumanEval repo and install in editable mode
RUN git clone https://github.com/openai/human-eval.git /workspace/human-eval && \
    pip install --no-cache-dir -e /workspace/human-eval


CMD ["bash"]
