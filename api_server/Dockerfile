FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Optional: use a Python version of your choice
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies + Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        wget curl git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Optional: Install Chrome and ChromeDriver (Uncomment if needed) ---
# Be aware: This will significantly increase your Docker image size.
# Ensure you really need a full browser and WebDriver inside this container.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     fonts-liberation \
#     libasound2 \
#     libatk-bridge2.0-0 \
#     libatk1.0-0 \
#     libc6 \
#     libcairo2 \
#     libcups2 \
#     libdbus-1-3 \
#     libexpat1 \
#     libfontconfig1 \
#     libgbm1 \
#     libgcc1 \
#     libglib2.0-0 \
#     libgtk-3-0 \
#     libnspr4 \
#     libnss3 \
#     libpango-1.0-0 \
#     libpangocairo-1.0-0 \
#     libstdc++6 \
#     libx11-6 \
#     libx11-xcb1 \
#     libxcb1 \
#     libxcomposite1 \
#     libxcursor1 \
#     libxdamage1 \
#     libxext6 \
#     libxfixes3 \
#     libxi6 \
#     libxrandr2 \
#     libxrender1 \
#     libxss1 \
#     libxtst6 \
#     lsb-release \
#     && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
#     && sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
#     && apt-get update \
#     && apt-get install -y google-chrome-stable \
#     # Install ChromeDriver
#     # Dynamically fetch the latest stable ChromeDriver version compatible with the installed Chrome
#     && CHROME_VERSION=$(google-chrome --version | cut -d' ' -f3 | cut -d'.' -f1) \
#     && CHROME_DRIVER_VERSION=$(wget -qO- "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION") \
#     && echo "Installing ChromeDriver version: $CHROME_DRIVER_VERSION for Chrome $CHROME_VERSION" \
#     && wget -q --continue -P /tmp "https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip" \
#     && unzip /tmp/chromedriver_linux64.zip -d /usr/local/bin \
#     && rm /tmp/chromedriver_linux64.zip \
#     && chmod +x /usr/local/bin/chromedriver \
#     # Clean up
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# --- End Optional Chrome/ChromeDriver Section ---

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This leverages Docker's build cache: if requirements.txt doesn't change,
# the pip install step won't re-run.
COPY api_server/requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir: Prevents pip from storing downloaded packages in a cache.
# --break-system-packages: Allows pip to install into system directories,
#                          necessary on some Debian-based systems when not using a venv.
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy the application code and necessary directories
# Assumes the Docker build context is the parent directory (e.g., your project root)
# Ensure these paths are correct relative to your build context.
COPY ./api_server /app/
COPY ./TextToImage/utils/ /app/TextToImage/utils/
COPY ./TextToImage/model/checkpoint/ /app/TextToImage/model/checkpoint/

# Create output directory
RUN mkdir -p /app/TextToImage/output

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application
# Note: Ensure 'model.py' is the correct entry point and handles all necessary paths.
# The OPENAI_API_KEY should be passed during 'docker run' (e.g., -e OPENAI_API_KEY="your_key").
CMD ["python", "model.py"]