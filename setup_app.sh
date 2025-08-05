#!/bin/bash
set -e

# Make sure we're running as 'admin'
if [ "$(whoami)" != "admin" ]; then
  echo "❌ This script must be run as user 'admin'."
  exit 1
fi

echo "🔧 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "📦 Installing required packages..."
sudo apt install -y git curl ca-certificates gnupg lsb-release apt-transport-https

echo "🐳 Installing Docker Engine..."
# Add Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker packages
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "✅ Docker installed successfully."

echo "🔧 Adding user 'admin' to docker group..."
sudo usermod -aG docker admin

echo "🛠 Enabling and starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker

echo "🔄 Refreshing group permissions (you may need to re-login)..."
newgrp docker <<EONG