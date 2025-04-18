!/bin/bash

if [!command -v "sudo" &> /dev/null]; then
    apt-get sudo || sudo apt-get update || sudo apt-get upgrade
fi

if [!command -v "curl" &> /dev/null]; then
    sudo apt-get install curl
fi

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22