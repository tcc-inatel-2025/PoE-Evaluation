FROM python:3.7-slim

# Install basic utilities
RUN apt-get update && \
    apt-get install -y git nano curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy your project
COPY ./human-eval /workspace/human-eval

# Install Python package in editable mode
RUN pip install --no-cache-dir -e /workspace/human-eval
RUN pip install requests

# Optional: expose a folder for any shared data (not strictly needed)
VOLUME /workspace/models
