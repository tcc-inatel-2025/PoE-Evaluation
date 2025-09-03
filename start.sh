#!/bin/bash

# Start Ollama in the background
ollama serve &

# Save PID (optional, for graceful shutdown)
OLLAMA_PID=$!

# Keep container alive (or run other commands)
tail -f /dev/null
