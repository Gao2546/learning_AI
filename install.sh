#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Function for Package Installation ---
# Usage: ensure_pkg_installed <command_to_check> [package_name]
# If package_name is not provided, it defaults to command_to_check
ensure_pkg_installed() {
    local cmd_to_check="$1"
    local pkg_name="${2:-$1}" # Use command name as package name if not specified
    local install_cmd=""

    # Check if command exists
    if ! command -v "$cmd_to_check" &> /dev/null; then
        echo "Command '$cmd_to_check' not found. Attempting to install package '$pkg_name'..."

        # Determine if we need/can use sudo
        if [[ $EUID -eq 0 ]]; then
            install_cmd="apt-get update && apt-get install -y $pkg_name"
        elif command -v sudo &> /dev/null; then
            install_cmd="sudo apt-get update && sudo apt-get install -y $pkg_name"
        else
            echo "Error: Cannot install '$pkg_name'. Need to run as root or have sudo installed."
            exit 1
        fi

        # Attempt installation
        echo "Running: $install_cmd"
        if ! eval "$install_cmd"; then
             echo "Error: Failed to install '$pkg_name'."
             exit 1
        fi

        # Verify command again after installation attempt
        if ! command -v "$cmd_to_check" &> /dev/null; then
            echo "Error: Package '$pkg_name' installed, but command '$cmd_to_check' still not found."
            # Special case for pip potentially being pip3
            if [[ "$cmd_to_check" == "pip" ]] && command -v "pip3" &> /dev/null; then
                echo "Note: 'pip3' command found instead of 'pip'. Continuing..."
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
# Note: We don't check for sudo itself here, the function handles it.
ensure_pkg_installed curl
ensure_pkg_installed git
ensure_pkg_installed python3
ensure_pkg_installed pip python3-pip # Check for pip command, install python3-pip package
# Check for ca-certificates package status instead of command
if ! dpkg -s ca-certificates &> /dev/null; then
    echo "Package 'ca-certificates' not found. Attempting to install..."
     if [[ $EUID -eq 0 ]]; then
        apt-get update && apt-get install -y ca-certificates
    elif command -v sudo &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ca-certificates
    else
        echo "Error: Cannot install 'ca-certificates'. Need to run as root or have sudo installed."
        exit 1
    fi
    echo "'ca-certificates' installed successfully."
else
    echo "Package 'ca-certificates' is already installed."
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