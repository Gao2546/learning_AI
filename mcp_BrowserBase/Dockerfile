# Stage 1: Base image with Python and essential tools
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies: curl, gnupg, wget, ca-certificates, fonts, udev rules, chromium, chromium-driver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    wget \
    ca-certificates \
    fonts-liberation \
    libu2f-udev \
    chromium \
    chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm (using NodeSource)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and install Python requirements
# Inferred from model.py, model2.py, model3.py
RUN echo "flask" > requirements.txt && \
    echo "selenium" >> requirements.txt && \
    echo "webdriver-manager" >> requirements.txt && \
    echo "beautifulsoup4" >> requirements.txt && \
    echo "langchain" >> requirements.txt && \
    echo "langchain-huggingface" >> requirements.txt && \
    echo "langchain-openai" >> requirements.txt && \
    echo "duckduckgo-search" >> requirements.txt && \
    echo "fastapi" >> requirements.txt && \
    echo "uvicorn[standard]" >> requirements.txt && \
    echo "torch --index-url https://download.pytorch.org/whl/cpu" >> requirements.txt && \
    echo "numpy" >> requirements.txt && \
    echo "requests" >> requirements.txt && \
    echo "python-dotenv" >> requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
COPY mcp_BrowserBase/package.json mcp_BrowserBase/package-lock.json ./mcp_BrowserBase/
RUN cd mcp_BrowserBase && npm ci

# Copy application source code
# Copy TextToImage first as it's a dependency for Python scripts
COPY TextToImage ./TextToImage
COPY mcp_BrowserBase ./mcp_BrowserBase

# Build TypeScript code
RUN cd mcp_BrowserBase && npm run build

# Create startup script
RUN echo '#!/bin/bash' > start.sh && \
    echo 'set -e' >> start.sh && \
    echo '# Start Python backend (FastAPI with Uvicorn) in the background' >> start.sh && \
    echo 'echo "Starting Python backend..."' >> start.sh && \
    echo 'uvicorn mcp_BrowserBase.model3:app --host 0.0.0.0 --port 5001 &' >> start.sh && \
    echo 'PYTHON_PID=$!' >> start.sh && \
    echo '# Wait a few seconds for the backend to start' >> start.sh && \
    echo 'sleep 5' >> start.sh && \
    echo '# Start Node.js MCP server in the foreground' >> start.sh && \
    echo 'echo "Starting Node.js MCP server..."' >> start.sh && \
    echo 'node mcp_BrowserBase/build/index.js' >> start.sh && \
    echo '# Wait for Python process to exit if Node server finishes (optional)' >> start.sh && \
    echo 'wait $PYTHON_PID' >> start.sh

RUN chmod +x start.sh

# Expose the Python backend port
EXPOSE 5001

# Set the entrypoint to the startup script
ENTRYPOINT ["./start.sh"]