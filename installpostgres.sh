#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# set -u
# Cause pipelines to return the exit status of the last command that failed.
set -o pipefail

echo "--- PostgreSQL User Setup Script Started ---"

# --- Check for sudo ---
echo "Checking for sudo..."
if ! command -v sudo &> /dev/null; then
    echo "ERROR: 'sudo' command not found."
    echo "This script requires sudo to install packages and manage PostgreSQL."
    echo "Please install sudo first (e.g., run as root: apt update && apt install sudo) and add your user to the sudo group."
    exit 1
else
    echo "'sudo' command found."
    # Optional: Refresh sudo timestamp
    # sudo -v
fi

# --- Check for psql (PostgreSQL client) and install PostgreSQL if necessary ---
echo "Checking for psql..."
if ! command -v psql &> /dev/null; then
    echo "'psql' command not found. Attempting to install PostgreSQL..."
    # Update package lists before installing
    echo "Updating package lists..."
    if ! sudo apt-get update; then
        echo "ERROR: 'sudo apt-get update' failed. Please check your internet connection and apt configuration."
        exit 1
    fi
    # Install PostgreSQL server and client non-interactively
    # The 'postgresql' package usually includes server, client, and dependencies,
    # and typically starts the service automatically on Debian/Ubuntu.
    echo "Installing PostgreSQL..."
    if ! sudo apt-get install -y postgresql postgresql-client; then
        echo "ERROR: Failed to install 'postgresql'. Please install it manually."
        exit 1
    fi
    echo "'postgresql' package installed successfully."
    echo "Waiting a few seconds for the PostgreSQL service to potentially start..."
    sleep 5 # Give the service a moment to initialize after installation
else
    echo "'psql' command found. Assuming PostgreSQL server is installed and running."
fi

# --- Create the PostgreSQL user ---
DB_USER="athip"
DB_PASSWORD="123456" # <-- WARNING: Hardcoding passwords in scripts is insecure. Consider alternatives for production.

echo "Attempting to create PostgreSQL user '${DB_USER}'..."

# Check if the user already exists first to make the script idempotent
# psql options: -t (tuples only), -A (unaligned), -c (command)
USER_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'")

if [ "$USER_EXISTS" = "1" ]; then
    echo "PostgreSQL user '${DB_USER}' already exists. Skipping creation."
else
    echo "User '${DB_USER}' does not exist. Creating..."
    # Execute the CREATE USER command
    # Using a 'here document' can be slightly cleaner for multi-line or complex SQL
    if sudo -u postgres psql -c "CREATE USER \"${DB_USER}\" WITH PASSWORD '${DB_PASSWORD}';"; then
        echo "PostgreSQL user '${DB_USER}' created successfully."
    else
        echo "ERROR: Failed to create PostgreSQL user '${DB_USER}'."
        # Double-check if it exists now, in case of a race condition or previous error reporting issue
        POST_CREATE_CHECK=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'")
        if [ "$POST_CREATE_CHECK" != "1" ]; then
             exit 1 # Exit if creation failed and user *still* doesn't exist
        else
             echo "User '${DB_USER}' seems to exist now despite reported error during creation."
        fi
    fi
fi

# Optional: Grant privileges if needed (example)
# echo "Granting privileges to user '${DB_USER}' on a database (example)..."
# sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE your_database_name TO ${DB_USER};" || echo "WARN: Failed to grant privileges (database might not exist yet)."


echo "--- PostgreSQL User Setup Script Finished Successfully ---"