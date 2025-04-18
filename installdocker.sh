#!/bin/bash
# Add Docker's official GPG key:
if [!command -v "sudo" &> /dev/null]; then
    apt-get sudo || sudo apt-get update || apt-get upgrade
fi

if [!command -v "curl" &> /dev/null]; then
    sudo apt-get install curl
fi

if [!command -v "ca-certificates" &> /dev/null]; then
    sudo apt-get install ca-certificates
fi

if [!command -v "docker-compose" &> /dev/null]; then
    sudo apt-get install docker-compose
fi

# sudo install -m 0755 -d /etc/apt/keyrings
# sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
# sudo chmod a+r /etc/apt/keyrings/docker.asc

# # Add the repository to Apt sources:
# echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
#   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
#   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# sudo apt-get update

# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin