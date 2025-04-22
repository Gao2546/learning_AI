# Use an official Python runtime as a parent image
# FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set a placeholder for the API key - it should be provided at runtime
ENV OPENAI_API_KEY=""

# Install system dependencies including Chrome and ChromeDriver
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \
    ca-certificates \
    # fonts-liberation \
    # libasound2 \
    # libatk-bridge2.0-0 \
    # libatk1.0-0 \
    # libc6 \
    # libcairo2 \
    # libcups2 \
    # libdbus-1-3 \
    # libexpat1 \
    # libfontconfig1 \
    # libgbm1 \
    # libgcc1 \
    # libglib2.0-0 \
    # libgtk-3-0 \
    # libnspr4 \
    # libnss3 \
    # libpango-1.0-0 \
    # libpangocairo-1.0-0 \
    # libstdc++6 \
    # libx11-6 \
    # libx11-xcb1 \
    # libxcb1 \
    # libxcomposite1 \
    # libxcursor1 \
    # libxdamage1 \
    # libxext6 \
    # libxfixes3 \
    # libxi6 \
    # libxrandr2 \
    # libxrender1 \
    # libxss1 \
    # libxtst6 \
    # lsb-release \
    iproute2 \
    curl \
    wget \
    xdg-utils \
    # && apt-get install -y --no-install-recommends \
    # python3 \
    # python3-pip \
    # Link python to python3
    && ln -s /usr/bin/python3 /usr/bin/python \
    # Install Chrome
    # && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    # && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    # && apt-get update \
    # && apt-get install -y google-chrome-stable \
    # # Install ChromeDriver
    # # Attempt to fetch the latest stable ChromeDriver version for Linux
    # && CHROME_DRIVER_VERSION=$(wget -qO- https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$(google-chrome --version | cut -d' ' -f3 | cut -d'.' -f1)) \
    # && echo "Using ChromeDriver version: $CHROME_DRIVER_VERSION" \
    # && wget -q --continue -P /tmp "https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip" \
    # && unzip /tmp/chromedriver_linux64.zip -d /usr/local/bin \
    # && rm /tmp/chromedriver_linux64.zip \
    # && chmod +x /usr/local/bin/chromedriver \
    # Clean up
    && apt-get purge -y --auto-remove wget unzip gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# Assumes the Docker build context is the parent directory (/home/athip/psu/learning_AI)
COPY api_server/requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --system to avoid potential conflicts with user installs if any
# RUN pip3 install --upgrade pip --break-system-packages
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy the application code and necessary directories
# Assumes the Docker build context is the parent directory (/home/athip/psu/learning_AI)
COPY api_server/ /app/
COPY TextToImage/utils/ /app/TextToImage/utils/ 
COPY TextToImage/model/checkpoint/ /app/TextToImage/model/checkpoint/
RUN mkdir -p /app/TextToImage/output

# Make port 5001 available to the world outside this container
EXPOSE 5000

# Define the command to run the app
# Note: The script model.py might need adjustments for file paths (like .key and model checkpoints)
# to work correctly within the Docker container environment.
# The OPENAI_API_KEY should be passed during 'docker run'.
CMD ["python", "model.py"]
