FROM python:3.7-slim

# Install basic utilities
RUN apt-get update && \
    apt-get install -y git nano curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the HumanEval repository
RUN git clone https://github.com/openai/human-eval.git /workspace/human-eval

# Install Python packages
RUN pip install --no-cache-dir -e /workspace/human-eval && \
    pip install --no-cache-dir requests radon pylint pandas matplotlib seaborn

# Set default command
CMD ["bash"]