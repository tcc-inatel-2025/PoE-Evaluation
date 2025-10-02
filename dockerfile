FROM python:3.7-slim

# Install basic utilities
RUN apt-get update && \
    apt-get install -y git nano curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the HumanEval repository
RUN git clone https://github.com/openai/human-eval.git /workspace/human-eval

# Install Python package in editable mode
RUN pip install --no-cache-dir -e /workspace/human-eval
RUN pip install --no-cache-dir requests
RUN pip install radon
RUN pip install pylint

# Set default command
CMD ["bash"]
