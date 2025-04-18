#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# set -u
# Cause pipelines to return the exit status of the last command that failed.
set -o pipefail

echo "--- NVM and Node.js Setup Script Started ---"

# --- Check for sudo ---
echo "Checking for sudo..."
if ! command -v sudo &> /dev/null; then
    echo "ERROR: 'sudo' command not found."
    echo "This script requires sudo to install packages."
    echo "Please install sudo first (e.g., run as root: apt update && apt install sudo) and add your user to the sudo group."
    exit 1
else
    echo "'sudo' command found."
    # Optional: Refresh sudo timestamp here if needed, although apt-get will prompt anyway
    # sudo -v
fi

# --- Check for curl and install if necessary ---
echo "Checking for curl..."
if ! command -v curl &> /dev/null; then
    echo "'curl' command not found. Attempting to install..."
    # Update package lists before installing
    if ! sudo apt-get update; then
        echo "ERROR: 'sudo apt-get update' failed. Please check your internet connection and apt configuration."
        exit 1
    fi
    # Install curl non-interactively
    if ! sudo apt-get install -y curl; then
        echo "ERROR: Failed to install 'curl'. Please install it manually."
        exit 1
    fi
    echo "'curl' installed successfully."
else
    echo "'curl' command found."
fi

# --- Install NVM (Node Version Manager) ---
# Use a known stable nvm version (v0.40.2 seems non-existent, let's use v0.39.7 as an example)
# Check https://github.com/nvm-sh/nvm/releases for the latest stable version
NVM_VERSION="v0.39.7"
NVM_INSTALL_URL="https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh"
NVM_DIR="$HOME/.nvm"

echo "Checking if NVM is already installed in ${NVM_DIR}..."
if [ -d "${NVM_DIR}" ]; then
    echo "NVM directory already exists. Skipping NVM installation."
    echo "If you want to reinstall or update, remove '${NVM_DIR}' first."
else
    echo "Installing NVM version ${NVM_VERSION}..."
    # Download and execute the install script
    if ! curl -o- "${NVM_INSTALL_URL}" | bash; then
        echo "ERROR: NVM installation failed."
        exit 1
    fi
    echo "NVM installation script completed."
fi

# --- Source NVM script to make nvm command available in *this* script session ---
NVM_SCRIPT_PATH="${NVM_DIR}/nvm.sh"
echo "Attempting to source NVM script from ${NVM_SCRIPT_PATH}..."

# Check if the nvm script actually exists before trying to source it
if [ ! -f "${NVM_SCRIPT_PATH}" ]; then
   echo "ERROR: NVM script not found at '${NVM_SCRIPT_PATH}' after installation."
   echo "Something went wrong with the NVM installation process."
   exit 1
fi

# Source the nvm script. Use '.' which is equivalent to 'source'
# The '\' before '.' prevents potential alias conflicts for '.'
\. "${NVM_SCRIPT_PATH}" # This loads nvm into the current script's environment

# Verify nvm command is available (optional but good practice)
if ! command -v nvm &> /dev/null; then
    echo "ERROR: 'nvm' command is not available after sourcing. NVM setup failed."
    exit 1
fi
echo "NVM sourced successfully. 'nvm' command is now available for this script."

# --- Install Node.js version 22 using NVM ---
NODE_VERSION="22"
echo "Installing Node.js version ${NODE_VERSION} using nvm..."
if ! nvm install "${NODE_VERSION}"; then
    echo "ERROR: Failed to install Node.js version ${NODE_VERSION} using nvm."
    exit 1
fi

# Set the installed version as the default (optional)
echo "Setting Node.js version ${NODE_VERSION} as default..."
if ! nvm alias default "${NODE_VERSION}"; then
    echo "WARN: Failed to set Node.js ${NODE_VERSION} as default."
fi

# Verify Node installation (optional)
echo "Verifying Node.js installation..."
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    echo "Node version: $(node -v)"
    echo "npm version: $(npm -v)"
else
    echo "WARN: Node or npm command not found after installation attempt."
fi


echo "--- NVM and Node.js Setup Script Finished Successfully ---"
echo ""
echo "IMPORTANT:"
echo "NVM has been installed and Node.js ${NODE_VERSION} is ready."
echo "To use 'nvm' and 'node' in your current terminal session outside this script,"
echo "you may need to either:"
echo "  1. Close and reopen your terminal."
echo "  2. Or run: source ~/.bashrc  (or ~/.zshrc, ~/.profile etc. depending on your shell)"