#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions ---
command_exists() {
    command -v "$1" &> /dev/null
}

package_installed() {
    dpkg -s "$1" &> /dev/null
}

# --- Configuration ---
# List packages needed. Format: "type:name"
# type can be 'cmd' (checked with command -v) or 'pkg' (checked with dpkg -s)
# The actual package name for apt install follows the last colon if different
REQUIRED_ITEMS=(
    "cmd:curl:curl"
    "pkg:ca-certificates:ca-certificates"
    "cmd:docker-compose:docker-compose" # Installs V1 via apt - see note below
    # Add other prerequisites here if needed
)

# --- Determine Package Manager and Sudo ---
SUDO_CMD=""
if [[ "$(id -u)" -ne 0 ]]; then
    echo "Not running as root. Checking for sudo..."
    if ! command_exists sudo; then
        echo "ERROR: sudo command not found. Please install sudo (e.g., 'apt update && apt install sudo' as root) or run this script as root." >&2
        exit 1
    fi
    SUDO_CMD="sudo"
    echo "Using sudo for privileges."
else
    echo "Running as root."
fi

PKG_MGR=""
if command_exists apt; then
    PKG_MGR="apt"
elif command_exists apt-get; then
    PKG_MGR="apt-get"
else
    echo "ERROR: Neither 'apt' nor 'apt-get' found. Cannot proceed." >&2
    exit 1
fi
echo "Using '$PKG_MGR' as package manager."

# --- Check Prerequisites ---
PACKAGES_TO_INSTALL=()
NEEDS_UPDATE=false

echo "Checking prerequisites..."
for item in "${REQUIRED_ITEMS[@]}"; do
    IFS=':' read -r type name package_name <<< "$item"
    # Default package_name to name if not specified
    package_name=${package_name:-$name}

    found=true
    if [[ "$type" == "cmd" ]]; then
        if ! command_exists "$name"; then
            echo " - Command '$name' is missing."
            found=false
        else
             echo " - Command '$name' is present."
        fi
    elif [[ "$type" == "pkg" ]]; then
        if ! package_installed "$package_name"; then
            echo " - Package '$package_name' is missing."
            found=false
        else
            echo " - Package '$package_name' is present."
        fi
    else
        echo "WARN: Unknown check type '$type' for item '$name'."
    fi

    if ! $found; then
        PACKAGES_TO_INSTALL+=("$package_name")
        NEEDS_UPDATE=true # Need to update if anything is missing
    fi
done

# --- Install Missing Packages ---
if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
    echo "The following packages need to be installed: ${PACKAGES_TO_INSTALL[*]}"

    if [[ "$NEEDS_UPDATE" = true ]]; then
        echo "Updating package lists ($PKG_MGR update)..."
        $SUDO_CMD $PKG_MGR update || { echo "ERROR: Failed to update package lists." >&2; exit 1; }
    fi

    echo "Installing missing packages..."
    # shellcheck disable=SC2086 # We want word splitting for $SUDO_CMD
    $SUDO_CMD $PKG_MGR install -y "${PACKAGES_TO_INSTALL[@]}" || { echo "ERROR: Failed to install packages." >&2; exit 1; }

    echo "Packages installed successfully."
else
    echo "All prerequisites are already met."
fi

# --- Final Notes ---
echo ""
if command_exists docker-compose && ! command_exists docker || ! docker compose version &>/dev/null ; then
    echo "NOTE: The 'docker-compose' package installed via apt/apt-get is likely Docker Compose V1." >&2
    echo "      For modern Docker installations, consider using the 'docker compose' plugin (V2)." >&2
    echo "      See: https://docs.docker.com/compose/install/" >&2
fi

echo ""
echo "Script finished successfully."
exit 0