#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Determine Package Manager ---
PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""
CHECK_PKG_CMD=""
INSTALL_OPTS="-y" # Common option for non-interactive install

echo "Detecting package manager..."

if command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    UPDATE_CMD="dnf check-update" # dnf uses check-update before install implicitly often, but explicit doesn't hurt
    INSTALL_CMD="dnf install"
    CHECK_PKG_CMD="rpm -q"
    echo "Using dnf (Fedora/RHEL/CentOS Stream)."
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    UPDATE_CMD="yum check-update"
    INSTALL_CMD="yum install"
    CHECK_PKG_CMD="rpm -q"
    echo "Using yum (Older RHEL/CentOS)."
elif command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    UPDATE_CMD="apt-get update"
    INSTALL_CMD="apt-get install"
    CHECK_PKG_CMD="dpkg -s"
    echo "Using apt-get (Debian/Ubuntu)."
else
    echo "Error: Could not find a supported package manager (apt-get, dnf, or yum)." >&2
    exit 1
fi

# --- Helper Function for Package Installation ---
# Usage: ensure_pkg_installed <command_to_check> [package_name]
# If package_name is not provided, it defaults to command_to_check
ensure_pkg_installed() {
    local cmd_to_check="$1"
    local pkg_name="${2:-$1}" # Use command name as package name if not specified
    local full_install_cmd=""
    local sudo_prefix=""

    # Determine if we need/can use sudo
    if [[ $EUID -ne 0 ]]; then
        if command -v sudo &> /dev/null; then
            sudo_prefix="sudo "
        else
             echo "Warning: Running as non-root and sudo command not found. Installations might fail."
             # Allow to proceed, maybe permissions are already sufficient or user knows what they are doing
        fi
    fi

    # Check if command exists
    if ! command -v "$cmd_to_check" &> /dev/null; then
        echo "Command '$cmd_to_check' not found. Attempting to install package '$pkg_name' using $PKG_MANAGER..."

        # Construct the installation command
        # For apt, update should usually run first. dnf/yum handle this more implicitly or via check-update.
        if [[ "$PKG_MANAGER" == "apt" ]]; then
            full_install_cmd="${sudo_prefix}${UPDATE_CMD} && ${sudo_prefix}${INSTALL_CMD} ${INSTALL_OPTS} ${pkg_name}"
        else
            # For dnf/yum, running check-update first is optional but can be good practice
            # full_install_cmd="${sudo_prefix}${UPDATE_CMD} > /dev/null 2>&1; ${sudo_prefix}${INSTALL_CMD} ${INSTALL_OPTS} ${pkg_name}"
            # Simpler: let dnf/yum handle dependencies/updates during install
            full_install_cmd="${sudo_prefix}${INSTALL_CMD} ${INSTALL_OPTS} ${pkg_name}"
        fi

        # Attempt installation
        echo "Running: ${full_install_cmd}"
        if ! eval "$full_install_cmd"; then
             echo "Error: Failed to install '$pkg_name' using $PKG_MANAGER."
             exit 1
        fi

        # Verify command again after installation attempt
        # Add a small delay in case path needs updating (rarely needed but can help)
        sleep 1
        hash -r 2>/dev/null || true # Reset bash's command lookup cache

        if ! command -v "$cmd_to_check" &> /dev/null; then
            echo "Error: Package '$pkg_name' installed via $PKG_MANAGER, but command '$cmd_to_check' still not found."
            # Special case for pip potentially being pip3
            if [[ "$cmd_to_check" == "pip" ]] && command -v "pip3" &> /dev/null; then
                echo "Note: 'pip3' command found instead of 'pip'. Assuming pip3 is sufficient. Continuing..."
            else
              exit 1
            fi
        fi
        echo "'$pkg_name' installation successful (command '$cmd_to_check' is available)."
    else
        echo "Command '$cmd_to_check' is already available."
    fi
}

# --- Check and Install Prerequisites ---
echo "Checking prerequisites..."

# Package names are often the same, but python-pip differs
PYTHON_PIP_PKG="python3-pip"
if [[ "$PKG_MANAGER" == "dnf" || "$PKG_MANAGER" == "yum" ]]; then
    # On recent Fedora/RHEL, python3-pip is correct. Older might just be python-pip?
    # Let's stick with python3-pip as it's standard for Python 3
    : # No change needed currently, python3-pip is usually correct
fi

ensure_pkg_installed curl
ensure_pkg_installed git
ensure_pkg_installed python3
ensure_pkg_installed pip "$PYTHON_PIP_PKG" # Check for pip command, install python3-pip package

# Check for ca-certificates package status
CA_CERTS_PKG="ca-certificates"
echo "Checking for package '$CA_CERTS_PKG'..."
ca_check_cmd="${CHECK_PKG_CMD} ${CA_CERTS_PKG} &> /dev/null"
needs_ca_install=false
if ! eval "$ca_check_cmd"; then
    needs_ca_install=true
    echo "Package '$CA_CERTS_PKG' not found."
fi

if [[ "$needs_ca_install" == "true" ]]; then
    echo "Attempting to install '$CA_CERTS_PKG' using $PKG_MANAGER..."
    sudo_prefix=""
     if [[ $EUID -ne 0 ]]; then
        if command -v sudo &> /dev/null; then
            sudo_prefix="sudo "
        else
            echo "Error: Cannot install '$CA_CERTS_PKG'. Need to run as root or have sudo installed."
            exit 1
        fi
    fi

    # Construct the installation command
    if [[ "$PKG_MANAGER" == "apt" ]]; then
        full_install_cmd="${sudo_prefix}${UPDATE_CMD} && ${sudo_prefix}${INSTALL_CMD} ${INSTALL_OPTS} ${CA_CERTS_PKG}"
    else
        full_install_cmd="${sudo_prefix}${INSTALL_CMD} ${INSTALL_OPTS} ${CA_CERTS_PKG}"
    fi

    echo "Running: $full_install_cmd"
    if ! eval "$full_install_cmd"; then
        echo "Error: Failed to install '$CA_CERTS_PKG' using $PKG_MANAGER."
        exit 1
    fi
    echo "'$CA_CERTS_PKG' installed successfully."
else
    echo "Package '$CA_CERTS_PKG' is already installed."
fi
echo "Prerequisite check complete."
echo # Newline for readability

# --- Argument Parsing ---
INSTALL_COMPONENT=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -i|--install)
      # Check if the value ($2) exists
      if [[ -z "$2" ]]; then
        echo "Error: Option '$1' requires an argument." >&2
        exit 1
      fi
      INSTALL_COMPONENT="$2"
      shift # past argument (-i)
      shift # past value (e.g., "all")
      ;;
    --) # End of options marker
      shift # Remove the --
      break # Stop processing options
      ;;
    -*)
      # Unknown option
      echo "Error: Unknown option '$1'" >&2
      exit 1
      ;;
    *)
      # Handle unexpected positional arguments if necessary
      echo "Error: Unexpected argument '$1'" >&2
      exit 1
      ;;
  esac
done

# --- Perform Installation Based on Argument ---
if [[ -n "$INSTALL_COMPONENT" ]]; then
    echo "Processing installation for component: $INSTALL_COMPONENT"
    case "$INSTALL_COMPONENT" in
        all)
            echo "Installing all components..."
            # Use . ./scriptname.sh to source from current dir explicitly
            if [ -f ./installenv.sh ]; then . ./installenv.sh; else echo "Warning: ./installenv.sh not found."; fi
            if [ -f ./installdocker.sh ]; then . ./installdocker.sh; else echo "Warning: ./installdocker.sh not found."; fi
            if [ -f ./installpostgres.sh ]; then . ./installpostgres.sh; else echo "Warning: ./installpostgres.sh not found."; fi
            if [ -f ./installnode.sh ]; then . ./installnode.sh; else echo "Warning: ./installnode.sh not found."; fi
            echo "Installation of all components attempted."
            ;;
        env)
            echo "Installing env component..."
            if [ -f ./installenv.sh ]; then . ./installenv.sh; else echo "Error: ./installenv.sh not found."; exit 1; fi
            ;;
        docker)
             echo "Installing docker component..."
            if [ -f ./installdocker.sh ]; then . ./installdocker.sh; else echo "Error: ./installdocker.sh not found."; exit 1; fi
            ;;
        postgres)
             echo "Installing postgres component..."
            if [ -f ./installpostgres.sh ]; then . ./installpostgres.sh; else echo "Error: ./installpostgres.sh not found."; exit 1; fi
            ;;
        node)
             echo "Installing node component..."
            if [ -f ./installnode.sh ]; then . ./installnode.sh; else echo "Error: ./installnode.sh not found."; exit 1; fi
            ;;
        *)
            echo "Error: Unknown installation component '$INSTALL_COMPONENT'" >&2
            echo "Available components: all, env, docker, postgres, node"
            exit 1
            ;;
    esac
else
    echo "No installation component specified. Use -i or --install option (e.g., -i all)."
    # Optionally exit with an error if the -i flag is mandatory
    # exit 1
fi

echo "Script finished."