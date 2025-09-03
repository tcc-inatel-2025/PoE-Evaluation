FROM python:3.7-slim

# Install basic utilities
RUN apt-get update && \
    apt-get install -y git nano curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy your project
COPY ./human-eval /workspace/human-eval

# Install Python package
RUN pip install --no-cache-dir -e /workspace/human-eval

RUN curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
RUN ollama --version

# Optional: expose a folder for Ollama models
VOLUME /workspace/models

# Copy startup script
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Expose Ollama default port
EXPOSE 11434

# Start container
CMD ["/workspace/start.sh"]