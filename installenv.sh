#!/bin/bash

if [!command -v "sudo" &> /dev/null]; then
    apt-get sudo || sudo apt-get update || sudo apt-get upgrade
else
    sudo apt-get update || sudo apt-get upgrade
fi

# git clone https://github.com/Gao2546/learning_AI.git
# cd learning_AI


# Create a directory named .env and change into it
if [!pwd .env]; then
    mkdir -p .env
    cd .env
fi

# Create a Python virtual environment named pytorch
if [!pwd pytorch]; then
    python3 -m venv pytorch
    cd ..
fi

# Activate the virtual environment
source ./env/pytorch/bin/activate

# Install the requirements from requirements.txt
pip install -r requirement.txt

python -m ipykernel install --user --name=pytorch --display-name "Python (pytorch)"